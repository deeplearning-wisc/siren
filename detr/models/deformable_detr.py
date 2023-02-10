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

from .segmentation import (dice_loss, sigmoid_focal_loss, sigmoid_focal_loss_binary)
import copy
from .vmf import vMF, vMFLogPartition

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
        self.args = args
        hidden_dim = transformer.d_model

        self.object_embedding_loss = object_embedding_loss

        self.class_embed = nn.Linear(hidden_dim, num_classes)

        if self.object_embedding_loss:
            if obj_embedding_head == 'intermediate':
                last_channel_size = 2*hidden_dim
            elif obj_embedding_head == 'head':
                last_channel_size = hidden_dim//2
            self.feature_embed = MLP(hidden_dim, hidden_dim, last_channel_size, 2)

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.args.siren:
            self.center_project = nn.Sequential(nn.Linear(hidden_dim, self.args.project_dim),
                                                nn.ReLU(),
                                                nn.Linear(self.args.project_dim, self.args.project_dim)
                                                  )



        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
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

        if self.args.siren:
            self.learnable_kappa = nn.Linear(num_classes,1, bias=False).cuda()
            torch.nn.init.constant_(self.learnable_kappa.weight, self.args.learnable_kappa_init)
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

    def forward(self, samples, targets=None, epoch=None):
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
        return self.forward_samples(samples, targets)

    def forward_samples(self, samples, targets=None):
        features, pos = self.backbone(samples)
        srcs = []
        masks = []
        if self.objectness:
            dim_index = 1
        for l, feat in enumerate(features):
            src, mask = feat.decompose()

            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
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
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight#300,512
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, masks,
                                                                                                            pos,query_embeds)
        # hs : 6,1,300,256
        outputs_classes = []
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
            if self.args.siren:
                if lvl == hs.shape[0] - 1:
                    output_project_features.append(self.center_project(hs[lvl]))

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()#8,300,4
            outputs_classes.append(outputs_class)

            pen_features.append(hs[lvl])
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)



        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['pen_features'] = pen_features[-1]
        out['epoch'] = self.epoch
        out['cls_head'] = self.class_embed[-1]
        if self.args.siren:
            assert len(output_project_features) == 1
        if self.args.siren:
            out['project_features'] = output_project_features[-1]
            out['project_head'] = self.center_project
            out['learnable_kappa'] = self.learnable_kappa

        ##end##
        if self.object_embedding_loss:
            out['pred_features'] = self.feature_embed(hs[-1])

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            if self.object_embedding_loss:
                out['pred_features'] = outputs_features
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_objectness=None):#, outputs_features):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if outputs_objectness is not None:
            return [{'pred_logits': a, 'pred_boxes': b, 'pred_nc_logits': c}#, 'pred_features': c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_objectness[:-1])] #, outputs_features[:-1])]
        else:
            return [{'pred_logits': a, 'pred_boxes': b}  # , 'pred_features': c}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]  # , outputs_features[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, args=None):
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
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ce = torch.nn.CrossEntropyLoss()
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.prototypes = torch.zeros(self.num_classes,
                                                  self.args.project_dim).cuda()

    def weighted_vmf_loss(self, pred, weight_before_exp, target):
        center_adpative_weight = weight_before_exp.view(1,-1)
        pred = center_adpative_weight * pred.exp() / (
                (center_adpative_weight * pred.exp()).sum(-1)).unsqueeze(-1)
        loss  = -(pred[range(target.shape[0]), target] + 1e-6).log().mean()

        return loss


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        epoch = outputs['epoch']

        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])


        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0],
             src_logits.shape[1],
             src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        if self.args.siren and self.args.siren_evaluate:
            if len(idx[0]) != 0:
                id_samples = outputs['project_features'][idx]
                for index in range(len(target_classes_o)):
                    self.prototypes.data[target_classes_o[index]] = \
                        F.normalize(0.05 * F.normalize(id_samples[index], p=2, dim=-1) + \
                      0.95 * self.prototypes.data[target_classes_o[index]], p=2, dim=-1)

                cosine_logits = F.cosine_similarity(
                    self.prototypes.data.detach().unsqueeze(0).repeat(len(id_samples), 1, 1),
                        id_samples.unsqueeze(1).repeat(1, len(self.prototypes.data), 1), 2)

                weight_before_exp = \
                    vMFLogPartition.apply(self.args.project_dim,
                    F.relu(outputs['learnable_kappa'].weight.view(-1, 1)))
                weight_before_exp = weight_before_exp.exp()

                cosine_similarity_loss = self.weighted_vmf_loss(
                    cosine_logits * F.relu(outputs['learnable_kappa'].weight.view(1, -1)),
                    weight_before_exp,
                    target_classes_o)

                assert len(idx[0]) == len(target_classes_o)
                loss_ce = sigmoid_focal_loss(src_logits,
                                             target_classes_onehot,
                                             num_boxes,
                                             alpha=self.focal_alpha,
                                             gamma=2) * src_logits.shape[1]

                if epoch == self.args.epochs - 1:
                    np.save(self.args.output_dir + '/proto.npy',
                            self.prototypes.cpu().data.numpy())
                    np.save(self.args.output_dir + '/kappa.npy',
                            outputs['learnable_kappa'].weight.cpu().data.numpy())


                losses = {'loss_ce': loss_ce,
                          'loss_vmf': self.args.vmf_weight * cosine_similarity_loss}

            else:
                print(idx)
                loss_dummy = (outputs['project_head'](torch.zeros(1,256).cuda())- \
                              outputs['project_head'](torch.zeros(1,256).cuda()))**2

                loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                             alpha=self.focal_alpha,
                                             gamma=2) * src_logits.shape[1]
                loss_dummy_lk = (outputs['learnable_kappa'](
                    torch.zeros(1, self.num_classes).cuda()) -
                                 outputs[
                    'learnable_kappa'](torch.zeros(
                                         1, self.num_classes).cuda())) ** 2

                losses = {'loss_ce': loss_ce,
                          'loss_vmf': loss_dummy.sum() \
                                         + loss_dummy_lk.sum()}


        else:
            loss_ce = sigmoid_focal_loss(src_logits,
                                         target_classes_onehot,
                                         num_boxes,
                                         alpha=self.focal_alpha,
                                         gamma=2) * src_logits.shape[1]
            losses = {'loss_ce': loss_ce}

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
        target_features = torch.stack([target_features[i][j] for i,j in zip(tgt_idx[0], tgt_idx[1])], dim=0)
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
        # print('hhh',target_classes_o)
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

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

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
        out_sampling_cls = None
        out_godin_h = None
        if 'project_features' in list(outputs.keys()):
            out_project_features = outputs['project_features']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_indexes = torch.nonzero(prob.reshape(-1) > 0.1).view(1, -1)
        scores = prob.reshape(-1)[topk_indexes[0]].view(1,-1)

        topk_boxes = topk_indexes // (out_logits.shape[2])
        labels = topk_indexes % (out_logits.shape[2])
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes1 = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        original_boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        new_ped_logits = torch.gather(out_logits, 1,
                                      topk_boxes.unsqueeze(-1).repeat(1,1, out_logits.shape[2]))
        out_pen_features = torch.gather(out_pen_features, 1,
                                        topk_boxes.unsqueeze(-1).repeat(1,1, out_pen_features.shape[2]))

        if 'project_features' in list(outputs.keys()):
            out_project_features = torch.gather(out_project_features,
                                                1, topk_boxes.unsqueeze(-1).repeat(1,1, out_project_features.shape[2]))


        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = boxes1 * scale_fct[:, None, :]


        results = [{'scores': s, 'labels': l, 'boxes': b, "original_boxes": ob, "logits_for_ood_eval": new_ped_logits,
                    "pen_features": out_pen_features,
                    "project_features": out_project_features,
                    "sampling_cls": out_sampling_cls,
                    'godin_h': out_godin_h,
                    "am_for_ood": None} \
                for s, l, b, ob in zip(scores, labels, boxes, original_boxes)]

        return results


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
        out_sampling_cls = None
        out_godin_h = None

        if 'project_features' in list(outputs.keys()):
            out_project_features = outputs['project_features']


        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        topk_values, topk_indexes = torch.topk(prob.reshape(out_logits.shape[0], -1), len(targets[0]['labels']), dim=1)
        scores = topk_values

        topk_boxes = topk_indexes // (out_logits.shape[2])
        labels = topk_indexes % (out_logits.shape[2])
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes1 = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        original_boxes = torch.gather(out_bbox, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        new_ped_logits = torch.gather(out_logits, 1, topk_boxes.unsqueeze(-1).repeat(1,1, out_logits.shape[2]))
        out_pen_features = torch.gather(out_pen_features, 1,
                                        topk_boxes.unsqueeze(-1).repeat(1, 1, out_pen_features.shape[2]))

        out_pen_features = torch.cat([out_pen_features, labels.unsqueeze(2)], -1)

        if 'project_features' in list(outputs.keys()):
            out_project_features = torch.gather(out_project_features,
                                                1, topk_boxes.unsqueeze(-1).repeat(1,1, out_project_features.shape[2]))
            out_project_features = torch.cat([out_project_features, labels.unsqueeze(2)], -1)


        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        boxes = boxes1 * scale_fct[:, None, :]


        results = [{'scores': s, 'labels': l, 'boxes': b, "original_boxes": ob, "logits_for_ood_eval": new_ped_logits,
                    "pen_features": out_pen_features,
                    "project_features": out_project_features,
                    "sampling_cls": out_sampling_cls,
                    'godin_h': out_godin_h,
                    "am_for_ood": None} \
                for s, l, b, ob in zip(scores, labels, boxes, original_boxes)]

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
