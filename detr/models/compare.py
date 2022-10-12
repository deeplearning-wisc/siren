# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .segmentation import (dice_loss, sigmoid_focal_loss)
import copy
from .vmf import VonMisesFisher


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, object_embedding_loss=False,
                 obj_embedding_head=None, objectness=None, args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.objectness = objectness
        self.transformer = transformer
        self.object_embedding_loss = object_embedding_loss
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        if objectness:
            self.objectness_layers = nn.Linear(hidden_dim, 2)
        if self.object_embedding_loss:
            if obj_embedding_head == 'intermediate':
                last_channel_size = 2 * hidden_dim
            elif obj_embedding_head == 'head':
                last_channel_size = hidden_dim // 2
            self.feature_embed = MLP(hidden_dim, hidden_dim, last_channel_size, 2)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.args = args
        if self.args.center_loss_scheme_project:
            self.center_project = nn.Linear(hidden_dim, self.args.project_dim)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if self.args.vmf:
            self.sampling_cls_layer = nn.Linear(self.args.project_dim, 2)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            # if self.object_embedding_loss:
            #     self.feature_embed = _get_clones(self.feature_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            if objectness:
                self.objectness_branch = nn.ModuleList([self.objectness_layers for _ in range(num_pred)])
            # if self.object_embedding_loss:
            #     self.feature_embed = nn.ModuleList([self.feature_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples, epoch=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        self.epoch = epoch
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        return self.forward_samples(samples)

    def forward_samples(self, samples):
        features, pos = self.backbone(samples)
        # breakpoint()
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        # breakpoint()
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
                # get the c6 features, masks and positional encodings.
        # breakpoint()
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight  # 300,512
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks,
                                                                                                            pos,
                                                                                                            query_embeds)
        # hs : 6,1,300,256
        outputs_classes = []
        output_objectness = []
        output_project_features = []
        pen_features = []
        outputs_coords = []
        outputs_features = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            if self.objectness:
                outputs_obj = self.objectness_branch[lvl](hs[lvl])
                output_objectness.append(outputs_obj)
            if self.args.center_loss_scheme_project and lvl == hs.shape[0] - 1:
                output_project_features.append(self.center_project(hs[lvl]))

            tmp = self.bbox_embed[lvl](hs[lvl])

            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()  # 8,300,4
            outputs_classes.append(outputs_class)
            pen_features.append(hs[lvl])
            outputs_coords.append(outputs_coord)
            # if self.object_embedding_loss:
            #     outputs_features.append(outputs_feat)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)
        # if self.object_embedding_loss:
        #     outputs_features = torch.stack(outputs_features)
        # breakpoint()
        if not self.objectness:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        else:
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
                   'objectness': output_objectness[-1]}
        out['backbone_features'] = features[-1].tensors
        out['pen_features'] = pen_features[-1]
        out['epoch'] = self.epoch
        out['cls_head'] = self.class_embed[-1]
        if self.args.vmf:
            out['sampling_cls_layer'] = self.sampling_cls_layer
        if self.args.center_loss_scheme_project:
            out['project_features'] = output_project_features[-1]
            out['project_head'] = self.center_project
        # breakpoint()

        ##end##
        if self.object_embedding_loss:
            out['pred_features'] = self.feature_embed(hs[-1])

        if self.aux_loss:
            if self.objectness:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, output_objectness,
                                                        out['backbone_features'])  # , outputs_features)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            if self.object_embedding_loss:
                out['pred_features'] = outputs_features
        # del out['backbone_features']
        # breakpoint()
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_objectness=None,
                      backbone_features=None):  # , outputs_features):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_objectness is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'objectness': c, 'backbone_features': backbone_features}
                    # , 'pred_features': c}
                    for a, b, c in
                    zip(outputs_class[:-1], outputs_coord[:-1], outputs_objectness[:-1])]  # , outputs_features[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b}  # , 'pred_features': c}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]  # , outputs_features[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, objectness, focal_alpha=0.25, args=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.empty_weight = torch.ones(self.num_classes).cuda()
        self.empty_weight[-1] = 0.1
        self.objectness = objectness

        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ce = torch.nn.CrossEntropyLoss()
        self.args = args
        # unknown
        if self.args.unknown or self.args.center_loss:
            # if self.args.unknown:
            self.start_epoch = self.args.unknown_start_epoch
            self.data_dict = torch.zeros(num_classes, self.args.sample_number, 256).cuda()
            self.number_dict = {}
            self.eye_matrix = torch.eye(256, device='cuda')
            for i in range(num_classes):
                self.number_dict[i] = 0
        if self.args.center_loss:
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
            if not self.args.center_loss_scheme_v1:
                if self.args.center_loss_scheme_project:
                    self.prototypes = torch.zeros(self.num_classes, self.args.project_dim).cuda()
                    if self.args.vmf:
                        self.start_epoch = self.args.unknown_start_epoch
                        self.data_dict = torch.zeros(num_classes, self.args.sample_number, self.args.project_dim).cuda()
                        self.number_dict = {}
                        for i in range(num_classes):
                            self.number_dict[i] = 0
                else:
                    self.prototypes = torch.zeros(self.num_classes,
                                                  256).cuda()  # torch.nn.Linear(256, self.num_classes).cuda()
            # self.register_buffer("prototypes", torch.zeros(self.num_classes, 256))

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        # assert 'pred_logits' in outputs
        # src_logits = outputs['pred_logits']
        #
        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # # breakpoint()
        # target_classes = torch.full(src_logits.shape[:2], self.num_classes-1,
        #                             dtype=torch.int64, device=src_logits.device)
        # target_classes[idx] = target_classes_o
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        #
        #
        # # multi-class focal loss.
        # # loss_ce1 = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction='none')
        # # pt = torch.exp(-loss_ce1)
        # # gamma = 2
        # # focal_loss = (self.focal_alpha * (1 - pt) ** gamma * loss_ce1).mean()
        # # # breakpoint()
        # # loss_ce = focal_loss
        # losses = {'loss_ce': loss_ce}
        # # breakpoint()
        # # if target_classes.max() == 20:
        # #     breakpoint()
        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        epoch = outputs['epoch']
        if not self.objectness:
            assert 'pred_logits' in outputs
            src_logits = outputs['pred_logits']

            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            # breakpoint()
            # loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
            # losses = {'loss_ce': loss_ce}

            # code for unknown-awareness
            if self.args.unknown:
                sum_temp = 0
                for index in range(self.num_classes):
                    sum_temp += self.number_dict[index]
                # breakpoint()
                if sum_temp == self.num_classes * self.args.sample_number and epoch < self.start_epoch:
                    gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                    id_samples = outputs['pen_features'][idx]
                    # maintaining an ID data queue for each class.
                    for index in range(len(gt_classes_numpy)):
                        dict_key = gt_classes_numpy[index]
                        self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                              id_samples[index].detach().view(1, -1)), 0)

                    # focal loss.
                    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                                 gamma=2) * src_logits.shape[1]
                    losses = {'loss_ce': loss_ce}

                elif sum_temp == self.num_classes * self.args.sample_number and epoch >= self.start_epoch:
                    gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                    id_samples = outputs['pen_features'][idx]
                    # maintaining an ID data queue for each class.
                    for index in range(len(gt_classes_numpy)):
                        dict_key = gt_classes_numpy[index]
                        self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                              id_samples[index].detach().view(1, -1)), 0)
                    # the covariance finder needs the data to be centered.
                    for index in range(self.num_classes):
                        if index == 0:
                            temp = self.data_dict[index].mean(0)
                            mean_embed_id = temp.view(1, -1)
                            X = self.data_dict[index] - temp
                        else:
                            temp = self.data_dict[index].mean(0)
                            X = torch.cat((X, self.data_dict[index] - temp), 0)
                            mean_embed_id = torch.cat((mean_embed_id,
                                                       temp.view(1, -1)), 0)

                    # add the variance.
                    temp_precision = torch.mm(X.t(), X) / len(X)
                    # for stable training.
                    temp_precision += 0.0001 * self.eye_matrix

                    for index in range(self.num_classes):
                        new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                            mean_embed_id[index], covariance_matrix=temp_precision)
                        negative_samples = new_dis.rsample((self.args.sample_from,))
                        prob_density = new_dis.log_prob(negative_samples)
                        # breakpoint()
                        # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                        # keep the data in the low density area.
                        cur_samples, index_prob = torch.topk(- prob_density, self.args.select)
                        if index == 0:
                            ood_samples = negative_samples[index_prob]
                        else:
                            ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                        del new_dis
                        del negative_samples
                    # new focal loss.
                    # breakpoint()

                    assert len(ood_samples) == self.num_classes * self.args.select
                    if not self.args.separate:
                        src_logits = torch.cat((src_logits, outputs['cls_head'](ood_samples).unsqueeze(0)), 1)
                        target_classes_onehot = torch.cat((target_classes_onehot,
                                                           torch.zeros((target_classes_onehot.shape[0],
                                                                        len(ood_samples),
                                                                        target_classes_onehot.shape[2])).cuda()), 1)
                        # print('hhh')
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]
                        losses = {'loss_ce': loss_ce}
                    else:
                        # pass
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]
                        loss_additional = self.args.separate_loss_weight * F.binary_cross_entropy_with_logits(
                            outputs['cls_head'](ood_samples),
                            torch.zeros((len(ood_samples), target_classes_onehot.shape[2])).cuda())
                        # breakpoint()
                        losses = {'loss_ce': loss_ce, 'loss_separate': loss_additional}

                    del ood_samples
                else:
                    gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                    id_samples = outputs['pen_features'][idx]
                    # maintaining an ID data queue for each class.
                    for index in range(len(gt_classes_numpy)):
                        dict_key = gt_classes_numpy[index]
                        if self.number_dict[dict_key] < self.args.sample_number:
                            self.data_dict[dict_key][self.number_dict[dict_key]] = id_samples[index].detach()
                            self.number_dict[dict_key] += 1

                    # focal loss.
                    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                                 gamma=2) * src_logits.shape[1]
                    losses = {'loss_ce': loss_ce}

            elif self.args.center_loss:
                if self.args.center_loss_scheme_v1:
                    sum_temp = 0
                    for index in range(self.num_classes):
                        sum_temp += self.number_dict[index]
                    # breakpoint()
                    if sum_temp == self.num_classes * self.args.sample_number and epoch < self.start_epoch:
                        gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                        id_samples = outputs['pen_features'][idx]
                        # maintaining an ID data queue for each class.
                        for index in range(len(gt_classes_numpy)):
                            dict_key = gt_classes_numpy[index]
                            self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                                  id_samples[index].detach().view(1, -1)), 0)

                        # focal loss.
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]
                        losses = {'loss_ce': loss_ce}

                    elif sum_temp == self.num_classes * self.args.sample_number and epoch >= self.start_epoch:
                        gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                        id_samples = outputs['pen_features'][idx]
                        # maintaining an ID data queue for each class.
                        for index in range(len(gt_classes_numpy)):
                            dict_key = gt_classes_numpy[index]
                            self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                                  id_samples[index].detach().view(1, -1)), 0)
                        # center_loss.
                        for index in range(self.num_classes):
                            if index == 0:
                                temp = self.data_dict[index].mean(0)
                                mean_embed_id = temp.view(1, -1)
                            else:
                                temp = self.data_dict[index].mean(0)
                                mean_embed_id = torch.cat((mean_embed_id,
                                                           temp.view(1, -1)), 0)
                        # breakpoint()
                        if len(idx[0]) != 0:
                            cosine_logits = F.cosine_similarity(
                                mean_embed_id.unsqueeze(0).repeat(len(id_samples), 1, 1),
                                id_samples.unsqueeze(1).repeat(1, len(mean_embed_id), 1), 2)
                            cosine_similarity_loss = self.criterion(cosine_logits, target_classes_o)
                            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                         alpha=self.focal_alpha,
                                                         gamma=2) * src_logits.shape[1]
                            # print(cosine_similarity_loss)
                            # breakpoint()
                            losses = {'loss_ce': loss_ce,
                                      'loss_center': self.args.center_weight * cosine_similarity_loss}
                        else:
                            print(idx)
                            # breakpoint()
                            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                         alpha=self.focal_alpha,
                                                         gamma=2) * src_logits.shape[1]
                            losses = {'loss_ce': loss_ce}
                    else:
                        gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                        id_samples = outputs['pen_features'][idx]
                        # maintaining an ID data queue for each class.
                        for index in range(len(gt_classes_numpy)):
                            dict_key = gt_classes_numpy[index]
                            if self.number_dict[dict_key] < self.args.sample_number:
                                self.data_dict[dict_key][self.number_dict[dict_key]] = id_samples[index].detach()
                                self.number_dict[dict_key] += 1

                        # focal loss.
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]
                        losses = {'loss_ce': loss_ce}
                elif self.args.center_loss_scheme_project:
                    if len(idx[0]) != 0:
                        id_samples = outputs['project_features'][idx]
                        for index in range(len(target_classes_o)):
                            if self.args.center_revise:
                                # breakpoint()
                                self.prototypes.data[target_classes_o[index]] = F.normalize(
                                    0.05 * F.normalize(id_samples[index], p=2, dim=-1) + \
                                    0.95 * self.prototypes.data[target_classes_o[index]], p=2, dim=-1)
                            else:
                                self.prototypes.data[target_classes_o[index]] = 0.05 * id_samples[index] + \
                                                                                0.95 * self.prototypes.data[
                                                                                    target_classes_o[index]]
                            # breakpoint()
                        cosine_logits = F.cosine_similarity(
                            self.prototypes.data.clone().detach().unsqueeze(0).repeat(len(id_samples), 1, 1),
                            id_samples.unsqueeze(1).repeat(1, len(self.prototypes.data), 1), 2)
                        # breakpoint()
                        cosine_similarity_loss = self.criterion(cosine_logits / self.args.center_temp, target_classes_o)

                        assert len(idx[0]) == len(target_classes_o)
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]
                        # breakpoint()
                        # print(cosine_logits)
                        if self.args.vmf:
                            lr_reg_loss = torch.tensor(0).cuda()
                            sum_temp = 0
                            for index in range(self.num_classes):
                                sum_temp += self.number_dict[index]
                            # breakpoint()
                            if sum_temp == self.num_classes * self.args.sample_number and epoch < self.start_epoch:
                                gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                                id_samples = F.normalize(outputs['project_features'][idx], dim=-1, p=2)
                                # maintaining an ID data queue for each class.
                                for index in range(len(gt_classes_numpy)):
                                    dict_key = gt_classes_numpy[index]
                                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                                          id_samples[index].detach().view(1, -1)), 0)

                            elif sum_temp == self.num_classes * self.args.sample_number and epoch >= self.start_epoch:
                                gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                                id_samples = F.normalize(outputs['project_features'][idx], dim=-1, p=2)
                                # maintaining an ID data queue for each class.
                                for index in range(len(gt_classes_numpy)):
                                    dict_key = gt_classes_numpy[index]
                                    self.data_dict[dict_key] = torch.cat((self.data_dict[dict_key][1:],
                                                                          id_samples[index].detach().view(1, -1)), 0)
                                # center_loss.
                                # for index in range(self.num_classes):
                                #     temp = self.data_dict[index].mean(0)
                                #     xm_norm = (temp ** 2).sum().sqrt()
                                #     # breakpoint()
                                #     mu0 = temp / xm_norm
                                #     # Full-batch ML estimator
                                #     kappa0 = (self.args.project_dim * xm_norm - xm_norm ** 3) / (1 - xm_norm ** 2)
                                #     vmf_true = VonMisesFisher(mu0.view(1,-1), kappa0.view(1,-1))
                                #     negative_samples = vmf_true.rsample(torch.Size([self.args.sample_from]))
                                #     prob_density = vmf_true.log_prob(negative_samples).view(-1)
                                #     # print(prob_density.shape, vmf_true.log_prob(negative_samples).shape)
                                #     # keep the data in the low density area.
                                #     cur_samples, index_prob = torch.topk(- prob_density, self.args.select)
                                #
                                #     if index == 0:
                                #         ood_samples = negative_samples.squeeze()[index_prob]
                                #     else:
                                #         ood_samples = torch.cat((ood_samples, negative_samples.squeeze()[index_prob]), 0)
                                #     del vmf_true
                                #     del negative_samples

                                # faster version.
                                temp = self.data_dict.mean(1)  # 20,16
                                xm_norm = (temp ** 2).sum(1).sqrt()  # 20
                                mu0 = temp / xm_norm.unsqueeze(1)
                                # Full-batch ML estimator
                                kappa0 = (self.args.project_dim * xm_norm - xm_norm ** 3) / (1 - xm_norm ** 2)
                                vmf_true = VonMisesFisher(mu0, kappa0.view(-1, 1))
                                # breakpoint()
                                # negative_samples = vmf_true.rsample(torch.Size([self.args.sample_from]))
                                # prob_density = vmf_true.log_prob(negative_samples)
                                # # keep the data in the low density area.
                                # cur_samples, index_prob = torch.topk(- prob_density, self.args.select, dim=0)
                                # for index in range(self.num_classes):
                                #     if index == 0:
                                #         ood_samples = negative_samples[index_prob[:, index], index, :]
                                #     else:
                                #         ood_samples = torch.cat((ood_samples, negative_samples[index_prob[:, index], index, :]), 0)
                                #
                                # del vmf_true
                                # del negative_samples
                                #
                                # # breakpoint()
                                # input_for_lr = torch.cat((outputs['sampling_cls_layer'](id_samples),
                                #                           outputs['sampling_cls_layer'](ood_samples)), 0)
                                # labels_for_lr = torch.cat((torch.ones(len(id_samples)).cuda(),
                                #                            torch.zeros(len(ood_samples)).cuda()), -1)
                                #
                                # criterion = torch.nn.CrossEntropyLoss()
                                # lr_reg_loss = criterion(input_for_lr, labels_for_lr.long())

                            else:
                                gt_classes_numpy = target_classes_o.int().cpu().data.numpy()
                                id_samples = F.normalize(outputs['project_features'][idx], dim=-1, p=2)
                                # maintaining an ID data queue for each class.

                                for index in range(len(gt_classes_numpy)):
                                    dict_key = gt_classes_numpy[index]
                                    if self.number_dict[dict_key] < self.args.sample_number:
                                        self.data_dict[dict_key][self.number_dict[dict_key]] = id_samples[
                                            index].detach()
                                        self.number_dict[dict_key] += 1
                                print(sum_temp)
                                print(idx)
                                print(self.number_dict)

                            loss_dummy = (outputs['sampling_cls_layer'](torch.zeros(1, 16).cuda()) - outputs[
                                'sampling_cls_layer'](
                                torch.zeros(1, 16).cuda())) ** 2
                            # print(loss_dummy.sum())
                            losses = {'loss_ce': loss_ce,
                                      'loss_center': self.args.center_weight * cosine_similarity_loss,
                                      'loss_vmf': self.args.vmf_weight * lr_reg_loss + loss_dummy.sum()}
                        else:
                            losses = {'loss_ce': loss_ce,
                                      'loss_center': self.args.center_weight * cosine_similarity_loss}
                    else:
                        print(idx)
                        # breakpoint()
                        loss_dummy = (outputs['project_head'](torch.zeros(1, 256).cuda()) - outputs['project_head'](
                            torch.zeros(1, 256).cuda())) ** 2
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]
                        print(loss_dummy)
                        losses = {'loss_ce': loss_ce, 'loss_center': loss_dummy.sum()}

                else:
                    if len(idx[0]) != 0:
                        id_samples = outputs['pen_features'][idx]
                        for index in range(len(target_classes_o)):
                            self.prototypes.data[target_classes_o[index]] = 0.05 * id_samples[index] + \
                                                                            0.95 * self.prototypes.data[
                                                                                target_classes_o[index]]
                        # breakpoint()
                        cosine_logits = F.cosine_similarity(
                            self.prototypes.data.clone().detach().unsqueeze(0).repeat(len(id_samples), 1, 1),
                            id_samples.unsqueeze(1).repeat(1, len(self.prototypes.data), 1), 2)
                        # breakpoint()
                        cosine_similarity_loss = self.criterion(cosine_logits / self.args.center_temp, target_classes_o)

                        assert len(idx[0]) == len(target_classes_o)
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]

                        losses = {'loss_ce': loss_ce,
                                  'loss_center': self.args.center_weight * cosine_similarity_loss}
                    else:
                        print(idx)
                        # breakpoint()
                        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                     alpha=self.focal_alpha,
                                                     gamma=2) * src_logits.shape[1]
                        losses = {'loss_ce': loss_ce}


            else:
                loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                             gamma=2) * src_logits.shape[1]
                losses = {'loss_ce': loss_ce}

            if log:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            return losses
        else:
            # breakpoint()
            assert 'pred_logits' in outputs
            src_logits = outputs['pred_logits']

            assert 'objectness' in outputs
            objectness_predictions = outputs['objectness']
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o

            ######
            out_bbox = outputs['pred_boxes']
            boxes1 = box_ops.box_cxcywh_to_xyxy(out_bbox)
            scale_fct_attention_map = torch.from_numpy(
                np.asarray([outputs['backbone_features'].shape[-1], outputs['backbone_features'].shape[-2],
                            outputs['backbone_features'].shape[-1],
                            outputs['backbone_features'].shape[-2]])).cuda().unsqueeze(0)
            # breakpoint()
            boxes_for_attention_map = boxes1 * scale_fct_attention_map[:, None, :].to(torch.float32)
            boxes_for_attention_map = boxes_for_attention_map.int()
            boxes_for_attention_map[boxes_for_attention_map < 0] = 0
            masks_attention_map = torch.zeros(
                (outputs['backbone_features'].shape[0], 300, outputs['backbone_features'].shape[2],
                 outputs['backbone_features'].shape[3])).cuda()

            attention_map = outputs['backbone_features'].mean(1).unsqueeze(1).repeat(1, 300, 1, 1)
            for image_index in range(outputs['backbone_features'].shape[0]):
                for index in range(300):
                    masks_attention_map[image_index][index][
                    boxes_for_attention_map[image_index][index][1]: boxes_for_attention_map[image_index][index][3],
                    boxes_for_attention_map[image_index][index][0]: boxes_for_attention_map[image_index][index][2]] = 1

            all_am_values = masks_attention_map.view(outputs['backbone_features'].shape[0], 300,
                                                     -1) * attention_map.view(outputs['backbone_features'].shape[0],
                                                                              300, -1)
            all_am_values = all_am_values.mean(-1)
            indices = torch.topk(all_am_values, 5, dim=-1)[1]

            batch_idx = torch.cat([torch.full_like(src, i) for i, src in enumerate(indices)])
            src_idx = torch.cat([src for src in indices])
            indices = (batch_idx, src_idx)
            target_classes[indices] = src_logits.shape[2] - 1

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

            #######
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            # breakpoint()
            loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes + len(indices[0]),
                                         alpha=self.focal_alpha, gamma=2) * \
                      src_logits.shape[1]

            # the objectness loss.
            objectness_classes = torch.full(objectness_predictions.shape[:2], 0, dtype=torch.int64,
                                            device=src_logits.device).cuda()
            indices_objectness = (torch.cat((indices[0], idx[0].cuda())), torch.cat((indices[1], idx[1].cuda())))
            # breakpoint()
            objectness_classes[indices_objectness] = 1
            objectness_classes_onehot = torch.zeros(
                [objectness_predictions.shape[0], objectness_predictions.shape[1], objectness_predictions.shape[2]],
                dtype=objectness_predictions.dtype, layout=objectness_predictions.layout,
                device=objectness_predictions.device)
            # breakpoint()
            objectness_classes_onehot.scatter_(2, objectness_classes.unsqueeze(-1), 1)
            loss_ce_obj = sigmoid_focal_loss(objectness_predictions, objectness_classes_onehot,
                                             len(indices_objectness[0]), alpha=self.focal_alpha,
                                             gamma=2) * objectness_predictions.shape[1]

            losses = {'loss_ce': loss_ce, 'loss_obj': loss_ce_obj * 0.1}  # 'dummy_loss': dummy_loss}

            if log:
                # TODO this should probably be a separate loss, not hacked in this one here
                losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss(self, z1, z2):  # BxNxD
        x = torch.einsum('ijd,icd->ijc', z1, z2)
        # x = torch.einsum('bnd,bnd->bnn', z1, z2)
        labels = torch.arange(0, x.shape[1]).view(1, x.shape[1]).repeat_interleave(x.shape[0], 0).cuda()
        # return self.ce(x[:, :1].squeeze(1), labels[:, :1].squeeze(1))
        return self.ce(x.flatten(0, 1), labels.flatten(0, 1))

    def align_tensor(self, t, indices):
        return torch.stack([t[b, indices[b].squeeze(-1)] for b in range(len(indices))])

    def loss_object_embedding(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_features = outputs['pred_features'][idx]
        tgt_idx = self._get_tgt_permutation_idx(indices)
        target_features = [t['patches'] for t in targets]
        target_features = torch.stack([target_features[i][j] for i, j in zip(tgt_idx[0], tgt_idx[1])], dim=0)
        return {'loss_object_embedding': torch.nn.functional.l1_loss(src_features, target_features, reduction='mean')}

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'object_embedding': self.loss_object_embedding
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def loss_labels_inter(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha,
                                     gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def get_loss_inter(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_inter,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'object_embedding': self.loss_object_embedding
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # breakpoint()
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # print(indices)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if loss == 'object_embedding':
                        continue
                    l_dict = self.get_loss_inter(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # breakpoint()
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_pen_features = outputs['pen_features']
        out_project_features = None
        if 'project_features' in list(outputs.keys()):
            out_project_features = outputs['project_features']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        # prob = F.softmax(out_logits, -1)
        # prob = prob[:, :, :-1]
        # breakpoint()
        # topk_values, topk_indexes = torch.topk(prob.reshape(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values
        # breakpoint()
        topk_indexes = torch.nonzero(prob.reshape(-1) > 0.1).view(1, -1)
        scores = prob.reshape(-1)[topk_indexes[0]].view(1, -1)

        # topk_boxes = topk_indexes // (out_logits.shape[2]-1)
        # labels = topk_indexes % (out_logits.shape[2]-1)
        topk_boxes = topk_indexes // (out_logits.shape[2])
        labels = topk_indexes % (out_logits.shape[2])
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes1 = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        original_boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        # breakpoint()
        # print(labels.max())
        # assert labels.max() < 20
        new_ped_logits = torch.gather(out_logits, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_logits.shape[2]))
        out_pen_features = torch.gather(out_pen_features, 1,
                                        topk_boxes.unsqueeze(-1).repeat(1, 1, out_pen_features.shape[2]))
        if 'project_features' in list(outputs.keys()):
            out_project_features = torch.gather(out_project_features,
                                                1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_project_features.shape[2]))
            # out_project_features = torch.cat([out_project_features, labels.unsqueeze(2)], -1)

        # breakpoint()
        # print(scores)
        # print(new_ped_logits[0].norm(dim=1).mean())
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # breakpoint()
        boxes = boxes1 * scale_fct[:, None, :]
        # scale_fct_attention_map = torch.from_numpy(np.asarray([outputs['backbone_features'].shape[-1], outputs['backbone_features'].shape[-2],
        #                                        outputs['backbone_features'].shape[-1], outputs['backbone_features'].shape[-2]])).cuda().unsqueeze(0)
        # boxes_for_attention_map = boxes1 * scale_fct_attention_map[:, None, :].to(torch.float32)
        # boxes_for_attention_map = boxes_for_attention_map.int()
        # boxes_for_attention_map[boxes_for_attention_map < 0] = 0
        # masks_attention_map = torch.zeros((100, outputs['backbone_features'].shape[2], outputs['backbone_features'].shape[3])).cuda()
        # # breakpoint()
        # attention_map = outputs['backbone_features'].mean(1).repeat(100, 1, 1)
        # for index in range(100):
        #     masks_attention_map[index][boxes_for_attention_map[0][index][1]: boxes_for_attention_map[0][index][3],
        #     boxes_for_attention_map[0][index][0]: boxes_for_attention_map[0][index][2]] = 1
        #
        # all_am_values = masks_attention_map.view(100, -1) * attention_map.view(100,-1)
        # all_am_values = all_am_values.mean(-1)

        # breakpoint()

        results = [{'scores': s, 'labels': l, 'boxes': b, "original_boxes": ob, "logits_for_ood_eval": new_ped_logits,
                    "pen_features": out_pen_features,
                    "project_features": out_project_features,
                    "am_for_ood": None} \
                   for s, l, b, ob in zip(scores, labels, boxes, original_boxes)]
        # breakpoint()
        return results

        # prob = out_logits.sigmoid()
        #
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values
        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        # # breakpoint()
        # new_ped_logits = torch.gather(out_logits, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_logits.shape[2]))
        #
        # # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]
        #
        # results = [{'scores': s, 'labels': l, 'boxes': b, "logits_for_ood_eval": new_ped_logits} for s, l, b in
        #            zip(scores, labels, boxes)]
        # # breakpoint()
        # return results

        # prob = F.softmax(out_logits, -1)
        # scores, labels = prob[..., :-1].max(-1)
        #
        # # convert to [x0, y0, x1, y1] format
        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # # and from relative [0, 1] to absolute [0, height] coordinates
        # img_h, img_w = target_sizes.unbind(1)
        # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]
        # # breakpoint()
        # results = [{'scores': s, 'labels': l, 'boxes': b, "logits_for_ood_eval": out_logits} for s, l, b in zip(scores, labels, boxes)]

    @torch.no_grad()
    def forward_maha(self, outputs, targets, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        out_pen_features = outputs['pen_features']
        out_project_features = None
        if 'project_features' in list(outputs.keys()):
            out_project_features = outputs['project_features']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        # prob = F.softmax(out_logits, -1)
        # prob = prob[:, :, :-1]
        # breakpoint()
        topk_values, topk_indexes = torch.topk(prob.reshape(out_logits.shape[0], -1), len(targets[0]['labels']), dim=1)
        scores = topk_values
        # breakpoint()
        # print(targets[0])
        # topk_indexes = torch.nonzero(prob.reshape(-1) > 0.1).view(1, -1)
        # scores = prob.reshape(-1)[topk_indexes[0]].view(1,-1)

        # topk_boxes = topk_indexes // (out_logits.shape[2]-1)
        # labels = topk_indexes % (out_logits.shape[2]-1)
        topk_boxes = topk_indexes // (out_logits.shape[2])
        labels = topk_indexes % (out_logits.shape[2])
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes1 = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        original_boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        # breakpoint()
        # print(labels.max())
        # assert labels.max() < 20
        new_ped_logits = torch.gather(out_logits, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_logits.shape[2]))
        out_pen_features = torch.gather(out_pen_features, 1,
                                        topk_boxes.unsqueeze(-1).repeat(1, 1, out_pen_features.shape[2]))
        # breakpoint()
        out_pen_features = torch.cat([out_pen_features, labels.unsqueeze(2)], -1)
        if 'project_features' in list(outputs.keys()):
            out_project_features = torch.gather(out_project_features,
                                                1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_project_features.shape[2]))
            out_project_features = torch.cat([out_project_features, labels.unsqueeze(2)], -1)
        # print(scores)
        # print(new_ped_logits[0].norm(dim=1).mean())
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # breakpoint()
        boxes = boxes1 * scale_fct[:, None, :]
        # scale_fct_attention_map = torch.from_numpy(np.asarray([outputs['backbone_features'].shape[-1], outputs['backbone_features'].shape[-2],
        #                                        outputs['backbone_features'].shape[-1], outputs['backbone_features'].shape[-2]])).cuda().unsqueeze(0)
        # boxes_for_attention_map = boxes1 * scale_fct_attention_map[:, None, :].to(torch.float32)
        # boxes_for_attention_map = boxes_for_attention_map.int()
        # boxes_for_attention_map[boxes_for_attention_map < 0] = 0
        # masks_attention_map = torch.zeros((100, outputs['backbone_features'].shape[2], outputs['backbone_features'].shape[3])).cuda()
        # # breakpoint()
        # attention_map = outputs['backbone_features'].mean(1).repeat(100, 1, 1)
        # for index in range(100):
        #     masks_attention_map[index][boxes_for_attention_map[0][index][1]: boxes_for_attention_map[0][index][3],
        #     boxes_for_attention_map[0][index][0]: boxes_for_attention_map[0][index][2]] = 1
        #
        # all_am_values = masks_attention_map.view(100, -1) * attention_map.view(100,-1)
        # all_am_values = all_am_values.mean(-1)

        # breakpoint()

        results = [{'scores': s, 'labels': l, 'boxes': b, "original_boxes": ob, "logits_for_ood_eval": new_ped_logits,
                    "pen_features": out_pen_features,
                    "project_features": out_project_features,
                    "am_for_ood": None} \
                   for s, l, b, ob in zip(scores, labels, boxes, original_boxes)]
        # breakpoint()
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
