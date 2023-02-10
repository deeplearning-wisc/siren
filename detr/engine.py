# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.voc_eval import VocEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from datasets.selfdet import selective_search
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.plot_utils import plot_prediction
from matplotlib import pyplot as plt
from visualize import visualize_prediction_results
import datasets.transforms as T

def train_one_epoch(model: torch.nn.Module, swav_model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # breakpoint()

        outputs = model(samples, epoch=epoch)
        if swav_model is not None:
            with torch.no_grad():
                for elem in targets:
                    elem['patches'] = swav_model(elem['patches'])
        if args.objectness:
            loss_dict = criterion(samples, outputs, targets)
        else:
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, dataset, args):
    model.eval()
    original_siren_flag = criterion.args.siren_evaluate
    if original_siren_flag:
        criterion.args.siren_evaluate = False

    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = (CocoEvaluator if 'COCO' in type(base_ds).__name__ else VocEvaluator)(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if 'loaddet' in output_dir:
            outputs = dict(pred_logits=[], pred_boxes=[])
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                loaded = torch.load(str(image_id) + '.pt', map_location=device)
                outputs['pred_logits'].append(loaded['pred_logits'])
                outputs['pred_boxes'].append(loaded['pred_boxes'])
            model = lambda *ignored: outputs

        outputs = model(samples)

        if args.objectness:
            loss_dict = criterion(samples, outputs, targets)
        else:
            loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if dataset == 'bdd':
            results[0]['labels'] += 1
        if 'savedet' in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                pred_logits = outputs['pred_logits'][i]
                pred_boxes = outputs['pred_boxes'][i]
                img_h, img_w = target['orig_size']
                pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.stack([img_w, img_h, img_w, img_h], dim=-1)
                torch.save(dict(image_id=image_id, target=target, pred_logits=pred_logits, pred_boxes=pred_boxes,
                                pred_boxes_=pred_boxes_), os.path.join(output_dir, str(image_id) + '.pt'))

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    if original_siren_flag:
        criterion.args.siren_evaluate = True

    return stats, coco_evaluator


@torch.no_grad()
def evaluate_ood_id(args, model, criterion, postprocessors, data_loader, base_ds, device, output_dir, address, dataset, vis_prediction_results):
    model.eval()
    original_siren_flag = criterion.args.siren_evaluate
    if original_siren_flag:
        criterion.args.siren_evaluate = False
    criterion.eval()
    all_logits = None
    pen_features = None
    project_features = None
    sampling_cls = None
    out_godin_h = None

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # breakpoint()
    coco_evaluator = (CocoEvaluator if 'COCO' in type(base_ds).__name__ else VocEvaluator)(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    index_cur = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if dataset == 'bdd':
            if index_cur > 20000:
                break
        else:
            if index_cur > 10000:
                break
        index_cur += 1
        if 'loaddet' in output_dir:
            outputs = dict(pred_logits = [], pred_boxes = [])
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                loaded = torch.load(str(image_id) + '.pt', map_location = device)
                outputs['pred_logits'].append(loaded['pred_logits'])
                outputs['pred_boxes'].append(loaded['pred_boxes'])
            model = lambda *ignored: outputs

        outputs = model(samples)

        if args.objectness:
            loss_dict = criterion(samples, outputs, targets)
        else:
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if args.maha_train:
            results = postprocessors['bbox'].forward_maha(outputs, targets, orig_target_sizes)
        else:
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        # breakpoint()
        if results[0]['logits_for_ood_eval'].shape[1] == 0:
            continue
        if dataset == 'bdd':
            results[0]['labels'] += 1
        import numpy as np
        # print(results[0]['logits_for_ood_eval'].shape)
        if all_logits is None:
            all_logits = results[0]['logits_for_ood_eval'].view(-1,
                                                                results[0]['logits_for_ood_eval'].shape[2]).cpu().data.numpy()
        else:
            temp = results[0]['logits_for_ood_eval'].view(-1,
                                                                results[0]['logits_for_ood_eval'].shape[2]).cpu().data.numpy()

            all_logits = np.concatenate((all_logits, temp), 0)


        # print(results[0]['logits_for_ood_eval'].shape)
        if pen_features is None:
            pen_features = results[0]['pen_features'].view(-1,
                                                                results[0]['pen_features'].shape[2]).cpu().data.numpy()
        else:
            temp = results[0]['pen_features'].view(-1,
                                                                results[0]['pen_features'].shape[2]).cpu().data.numpy()

            pen_features = np.concatenate((pen_features, temp), 0)

        if results[0]['project_features'] is not None:
            if project_features is None:
                project_features = results[0]['project_features'].view(-1,
                                                      results[0]['project_features'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['project_features'].view(-1,
                                                      results[0]['project_features'].shape[2]).cpu().data.numpy()

                project_features = np.concatenate((project_features, temp), 0)


        if results[0]['sampling_cls'] is not None:
            if sampling_cls is None:
                sampling_cls = results[0]['sampling_cls'].view(-1,
                                                      results[0]['sampling_cls'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['sampling_cls'].view(-1,
                                                      results[0]['sampling_cls'].shape[2]).cpu().data.numpy()

                sampling_cls = np.concatenate((sampling_cls, temp), 0)

        if results[0]['godin_h'] is not None:
            if out_godin_h is None:
                out_godin_h = results[0]['godin_h'].view(-1,
                                                      results[0]['godin_h'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['godin_h'].view(-1,
                                                      results[0]['godin_h'].shape[2]).cpu().data.numpy()

                out_godin_h = np.concatenate((out_godin_h, temp), 0)

        # all_logits.append(results[0]['logits_for_ood_eval'].squeeze().cpu().data.numpy())
        # print(all_logits.shape)
        if vis_prediction_results:
            visualize_prediction_results(samples, results, output_dir, targets, False)
        if 'savedet' in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                pred_logits = outputs['pred_logits'][i]
                pred_boxes = outputs['pred_boxes'][i]
                img_h, img_w = target['orig_size']
                pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.stack([img_w, img_h, img_w, img_h], dim=-1)
                torch.save(dict(image_id=image_id, target=target, pred_logits=pred_logits, pred_boxes=pred_boxes,
                                pred_boxes_=pred_boxes_), os.path.join(output_dir, str(image_id) + '.pt'))

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # breakpoint()
    if not args.maha_train:
        np.save(address + 'id-logits.npy', all_logits)
        np.save(address + 'id-pen.npy', pen_features)
        if project_features is not None:
            np.save(address + 'id-pro.npy', project_features)
        if sampling_cls is not None:
            np.save(address + 'id-sampling.npy', sampling_cls)
        if out_godin_h is not None:
            np.save(address + 'id-godin.npy', out_godin_h)
    else:
        np.save(address + 'id-logits_maha_train.npy', all_logits)
        np.save(address + 'id-pen_maha_train.npy', pen_features)
        if project_features is not None:
            np.save(address + 'id-pro_maha_train.npy', project_features)
        if sampling_cls is not None:
            np.save(address + 'id-sampling_maha_train.npy', sampling_cls)
        if out_godin_h is not None:
            np.save(address + 'id-godin_maha_train.npy', out_godin_h)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


def evaluate_ood_ood(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, address, dataset, vis_prediction_results):
    model.eval()
    criterion.eval()
    original_siren_flag = criterion.args.siren_evaluate
    if original_siren_flag:
        criterion.args.siren_evaluate = False
    all_logits = None
    pen_features = None
    project_features = None
    sampling_cls = None
    out_godin_h = None

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # breakpoint()
    coco_evaluator = (CocoEvaluator if 'COCO' in type(base_ds).__name__ else VocEvaluator)(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in data_loader:#metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # breakpoint()
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if 'loaddet' in output_dir:
            outputs = dict(pred_logits=[], pred_boxes=[])
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                loaded = torch.load(str(image_id) + '.pt', map_location=device)
                outputs['pred_logits'].append(loaded['pred_logits'])
                outputs['pred_boxes'].append(loaded['pred_boxes'])
            model = lambda *ignored: outputs

        outputs = model(samples)
        # breakpoint()
        # if 'ood' in dataset:
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)


        if results[0]['logits_for_ood_eval'].shape[1] == 0:
            continue
        import numpy as np
        if all_logits is None:
            all_logits = results[0]['logits_for_ood_eval'].view(-1,
                                                                results[0]['logits_for_ood_eval'].shape[2]).cpu().data.numpy()
        else:
            temp = results[0]['logits_for_ood_eval'].view(-1,
                                                                results[0]['logits_for_ood_eval'].shape[2]).cpu().data.numpy()

            all_logits = np.concatenate((all_logits, temp), 0)


        if pen_features is None:
            pen_features = results[0]['pen_features'].view(-1,
                                                                results[0]['pen_features'].shape[2]).cpu().data.numpy()
        else:
            temp = results[0]['pen_features'].view(-1,
                                                                results[0]['pen_features'].shape[2]).cpu().data.numpy()

            pen_features = np.concatenate((pen_features, temp), 0)


        if results[0]['project_features'] is not None:
            if project_features is None:
                project_features = results[0]['project_features'].view(-1,
                                                      results[0]['project_features'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['project_features'].view(-1,
                                                      results[0]['project_features'].shape[2]).cpu().data.numpy()

                project_features = np.concatenate((project_features, temp), 0)

        if results[0]['sampling_cls'] is not None:
            if sampling_cls is None:
                sampling_cls = results[0]['sampling_cls'].view(-1,
                                                      results[0]['sampling_cls'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['sampling_cls'].view(-1,
                                                      results[0]['sampling_cls'].shape[2]).cpu().data.numpy()

                sampling_cls = np.concatenate((sampling_cls, temp), 0)

        if results[0]['godin_h'] is not None:
            if out_godin_h is None:
                out_godin_h = results[0]['godin_h'].view(-1,
                                                      results[0]['godin_h'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['godin_h'].view(-1,
                                                      results[0]['godin_h'].shape[2]).cpu().data.numpy()

                out_godin_h = np.concatenate((out_godin_h, temp), 0)

        # all_logits.append(results[0]['logits_for_ood_eval'].cpu().data.numpy())
        if vis_prediction_results:
            visualize_prediction_results(samples, results, output_dir, targets, True)
        continue
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        if 'savedet' in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                pred_logits = outputs['pred_logits'][i]
                pred_boxes = outputs['pred_boxes'][i]
                img_h, img_w = target['orig_size']
                pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.stack([img_w, img_h, img_w, img_h], dim=-1)
                torch.save(dict(image_id=image_id, target=target, pred_logits=pred_logits, pred_boxes=pred_boxes,
                                pred_boxes_=pred_boxes_), os.path.join(output_dir, str(image_id) + '.pt'))

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    if dataset == 'openimages_ood_val':
        np.save(address + 'ood-open-logits.npy', all_logits)
        np.save(address + 'ood-open-pen.npy', pen_features)
    else:
        np.save(address + 'ood-logits.npy', all_logits)
        np.save(address + 'ood-pen.npy', pen_features)
    if project_features is not None:
        if dataset == 'openimages_ood_val':
            np.save(address + 'ood-open-pro.npy', project_features)
        else:
            np.save(address + 'ood-pro.npy', project_features)
    if sampling_cls is not None:
        if dataset == 'openimages_ood_val':
            np.save(address + 'ood-open-sampling.npy', sampling_cls)
        else:
            np.save(address + 'ood-sampling.npy', sampling_cls)
    if out_godin_h is not None:
        if dataset == 'openimages_ood_val':
            np.save(address + 'ood-open-godin.npy', out_godin_h)
        else:
            np.save(address + 'ood-godin.npy', out_godin_h)
    return None, None
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator



@torch.no_grad()
def evaluate_ood_id_speckle(args, model, criterion, postprocessors, data_loader, base_ds, device, output_dir, address, dataset, vis_prediction_results):
    model.eval()
    original_unknown_flag = criterion.args.unknown
    original_center_flag = criterion.args.center_loss
    original_csi_flag = criterion.args.csi
    if original_unknown_flag:
        criterion.args.unknown = False
    if original_center_flag:
        criterion.args.center_loss = False
    if original_csi_flag:
        criterion.args.csi = False
        model.args.csi = False
    criterion.eval()
    all_logits = None
    pen_features = None
    project_features = None
    sampling_cls = None
    out_godin_h = None

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # breakpoint()
    coco_evaluator = (CocoEvaluator if 'COCO' in type(base_ds).__name__ else VocEvaluator)(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    index_cur = 0
    speckle = lambda x: (x + x * torch.randn_like(x.float()))
    # normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        if dataset == 'bdd':
            if index_cur > 20000:
                break
        else:
            if index_cur > 10000:
                break
        index_cur += 1
        if 'loaddet' in output_dir:
            outputs = dict(pred_logits = [], pred_boxes = [])
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                loaded = torch.load(str(image_id) + '.pt', map_location = device)
                outputs['pred_logits'].append(loaded['pred_logits'])
                outputs['pred_boxes'].append(loaded['pred_boxes'])
            model = lambda *ignored: outputs
        # breakpoint()
        samples.tensors = speckle(samples.tensors)
        outputs = model(samples)

        if args.objectness:
            loss_dict = criterion(samples, outputs, targets)
        else:
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if args.maha_train:
            results = postprocessors['bbox'].forward_maha(outputs, targets, orig_target_sizes)
        else:
            results = postprocessors['bbox'](outputs, orig_target_sizes)
        # breakpoint()
        if results[0]['logits_for_ood_eval'].shape[1] == 0:
            continue
        if dataset == 'bdd':
            results[0]['labels'] += 1
        import numpy as np
        # print(results[0]['logits_for_ood_eval'].shape)
        if all_logits is None:
            all_logits = results[0]['logits_for_ood_eval'].view(-1,
                                                                results[0]['logits_for_ood_eval'].shape[2]).cpu().data.numpy()
        else:
            temp = results[0]['logits_for_ood_eval'].view(-1,
                                                                results[0]['logits_for_ood_eval'].shape[2]).cpu().data.numpy()

            all_logits = np.concatenate((all_logits, temp), 0)


        # print(results[0]['logits_for_ood_eval'].shape)
        if pen_features is None:
            pen_features = results[0]['pen_features'].view(-1,
                                                                results[0]['pen_features'].shape[2]).cpu().data.numpy()
        else:
            temp = results[0]['pen_features'].view(-1,
                                                                results[0]['pen_features'].shape[2]).cpu().data.numpy()

            pen_features = np.concatenate((pen_features, temp), 0)

        if results[0]['project_features'] is not None:
            if project_features is None:
                project_features = results[0]['project_features'].view(-1,
                                                      results[0]['project_features'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['project_features'].view(-1,
                                                      results[0]['project_features'].shape[2]).cpu().data.numpy()

                project_features = np.concatenate((project_features, temp), 0)


        if results[0]['sampling_cls'] is not None:
            if sampling_cls is None:
                sampling_cls = results[0]['sampling_cls'].view(-1,
                                                      results[0]['sampling_cls'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['sampling_cls'].view(-1,
                                                      results[0]['sampling_cls'].shape[2]).cpu().data.numpy()

                sampling_cls = np.concatenate((sampling_cls, temp), 0)

        if results[0]['godin_h'] is not None:
            if out_godin_h is None:
                out_godin_h = results[0]['godin_h'].view(-1,
                                                      results[0]['godin_h'].shape[2]).cpu().data.numpy()
            else:
                temp = results[0]['godin_h'].view(-1,
                                                      results[0]['godin_h'].shape[2]).cpu().data.numpy()

                out_godin_h = np.concatenate((out_godin_h, temp), 0)

        # all_logits.append(results[0]['logits_for_ood_eval'].squeeze().cpu().data.numpy())
        # print(all_logits.shape)
        if vis_prediction_results:
            visualize_prediction_results(samples, results, output_dir, targets, False)
        if 'savedet' in output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, target in enumerate(targets):
                image_id = target['image_id'].item()
                pred_logits = outputs['pred_logits'][i]
                pred_boxes = outputs['pred_boxes'][i]
                img_h, img_w = target['orig_size']
                pred_boxes_ = box_cxcywh_to_xyxy(pred_boxes) * torch.stack([img_w, img_h, img_w, img_h], dim=-1)
                torch.save(dict(image_id=image_id, target=target, pred_logits=pred_logits, pred_boxes=pred_boxes,
                                pred_boxes_=pred_boxes_), os.path.join(output_dir, str(image_id) + '.pt'))

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # breakpoint()
    if not args.maha_train:
        np.save(address + 'id-logits-speckle.npy', all_logits)
        np.save(address + 'id-pen-speckle.npy', pen_features)
        if project_features is not None:
            np.save(address + 'id-pro-speckle.npy', project_features)
        if sampling_cls is not None:
            np.save(address + 'id-sampling-speckle.npy', sampling_cls)
        if out_godin_h is not None:
            np.save(address + 'id-godin-speckle.npy', out_godin_h)
    else:
        np.save(address + 'id-logits_maha_train-speckle.npy', all_logits)
        np.save(address + 'id-pen_maha_train-speckle.npy', pen_features)
        if project_features is not None:
            np.save(address + 'id-pro_maha_train-speckle.npy', project_features)
        if sampling_cls is not None:
            np.save(address + 'id-sampling_maha_train-speckle.npy', sampling_cls)
        if out_godin_h is not None:
            np.save(address + 'id-godin_maha_train-speckle.npy', out_godin_h)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


@torch.no_grad()
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        top_k = len(targets[0]['boxes'])

        outputs = model(samples)
        indices = outputs['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]
        predictied_boxes = torch.stack([outputs['pred_boxes'][0][i] for i in indices]).unsqueeze(0)
        logits = torch.stack([outputs['pred_logits'][0][i] for i in indices]).unsqueeze(0)
        fig, ax = plt.subplots(1, 3, figsize=(10,3), dpi=200)

        img = samples.tensors[0].cpu().permute(1,2,0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
        img = img.astype('uint8')
        h, w = img.shape[:-1]


        # SS results
        boxes_ss = get_ss_res(img, h, w, top_k)
        plot_prediction(samples.tensors[0:1], boxes_ss, torch.zeros(1, boxes_ss.shape[1], 4).to(logits), ax[0], plot_prob=False)
        ax[0].set_title('Selective Search')

        # Pred results
        plot_prediction(samples.tensors[0:1], predictied_boxes, logits, ax[1], plot_prob=False)
        ax[1].set_title('Prediction (Ours)')

        # GT Results
        plot_prediction(samples.tensors[0:1], targets[0]['boxes'].unsqueeze(0), torch.zeros(1, targets[0]['boxes'].shape[0], 4).to(logits), ax[2], plot_prob=False)
        ax[2].set_title('GT')

        for i in range(3):
            ax[i].set_aspect('equal')
            ax[i].set_axis_off()

        plt.savefig(os.path.join(output_dir, f'img_{int(targets[0]["image_id"][0])}.jpg'))


def get_ss_res(img, h, w, top_k):
    boxes = selective_search(img, h, w)[:top_k]
    boxes = torch.tensor(boxes).unsqueeze(0)
    boxes = box_xyxy_to_cxcywh(boxes)/torch.tensor([w, h, w, h])
    return boxes

