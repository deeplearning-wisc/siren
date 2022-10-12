import argparse
import numpy as np

default_settings = {
    'deformable_detr': {
        'lr': 2e-4,
        'lr_backbone': 2e-5,
        'epochs': 50,
        'lr_drop': 40,
        'dim_feedforward': 1024,
        'num_queries': 300,
        'set_cost_class': 2,
        'cls_loss_coef': 2
    },
    'detr': {
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'epochs': 300,
        'lr_drop': 200,
        'dim_feedforward': 2048,
        'num_queries': 100,
        'set_cost_class': 1,
        'cls_loss_coef': 1
    }
}


def set_model_defaults(args):
    defaults = default_settings[args.model]
    runtime_args = vars(args)
    for k, v in runtime_args.items():
        if v is None and k in defaults:
            setattr(args, k, defaults[k])
    return args


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--model', default='deformable_detr', type=str, choices=['detr', 'deformable_detr'])
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max_prop', default=30, type=int)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr_drop', type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--filter_pct', type=float, default=-1)
    parser.add_argument('--filter_num', type=float, default=-1)
    parser.add_argument('--reset_embedding_layer', type=int, default=1)

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--strategy', default='topk', type=str,
                        choices=['topk', 'topk_edgebox', 'mc_1', 'mc_2', 'mc_3', 'mc_4', 'random_sample', 'random'])
    parser.add_argument('--obj_embedding_head', default='intermediate', type=str, choices=['intermediate', 'head'])

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')
    parser.add_argument('--load_backbone', default='swav', type=str)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--object_embedding_coef', default=1, type=float,
                        help="object_embedding_coef box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--dataset', default='imagenet')
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--cache_path', default=None, help='where to store the cache')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--random_seed', action='store_true')
    parser.add_argument('--eval_every', default=1, type=int)
    parser.add_argument('--save_every', default=1, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--maha_train', action='store_true')
    parser.add_argument('--viz_prediction_results', action='store_true')
    parser.add_argument('--objectness', action='store_true')
    parser.add_argument('--unknown', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--object_embedding_loss', default=False, action='store_true', help='whether to use this loss')
    #detr
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    # unknown
    parser.add_argument('--sample_number', default=1000, type=int)
    parser.add_argument('--sample_from', default=10000, type=int)
    parser.add_argument('--unknown_start_epoch', default=30, type=int)
    parser.add_argument('--select', default=1, type=int)
    parser.add_argument('--separate', action='store_true')
    parser.add_argument('--separate_loss_weight', type=float, default=1.0)
    parser.add_argument('--center_loss', action='store_true')
    parser.add_argument('--center_loss_scheme_v1', default=0, type=int)
    parser.add_argument('--center_loss_scheme_project', default=0, type=int)
    parser.add_argument('--project_dim', default=128, type=int)
    parser.add_argument('--center_temp', default=0.1, type=float)
    parser.add_argument('--center_weight', type=float, default=1.0)
    parser.add_argument('--vmf', action='store_true')
    parser.add_argument('--vmf_weight', default=1.0, type=float)
    parser.add_argument('--vmf_imbalance', action='store_true')
    parser.add_argument('--vmf_add_sample', action='store_true')
    parser.add_argument('--vmf_multi_layer', action='store_true')
    parser.add_argument('--mlp_project', action='store_true')
    parser.add_argument('--add_class_prior', action='store_true')
    parser.add_argument('--special_lr', action='store_true')
    parser.add_argument('--on_the_pen', action='store_true')
    parser.add_argument('--on_the_pen_ours', action='store_true')
    parser.add_argument('--godin', action='store_true')
    parser.add_argument('--csi', action='store_true')
    parser.add_argument('--eval_speckle', action='store_true')
    parser.add_argument('--center_vmf_learnable_kappa', action='store_true')
    parser.add_argument('--center_vmf_fix_kappa', action='store_true')
    parser.add_argument('--center_vmf_no_kappa', action='store_true')
    parser.add_argument('--center_vmf_no_zp', action='store_true')
    parser.add_argument('--center_vmf_estimate_kappa', action='store_true')
    parser.add_argument('--learnable_kappa_init', type=float, default=10)
    parser.add_argument('--learnable_kappa_init_normal', action='store_true')
    parser.add_argument('--learnable_kappa_init_uniform', action='store_true')
    parser.add_argument('--vmf_multi_classifer_ml', action='store_true')
    parser.add_argument('--vmf_multi_classifer', action='store_true')
    parser.add_argument('--vmf_loss_single_bit', action='store_true')
    parser.add_argument('--vmf_focal_loss', action='store_true')
    parser.add_argument('--center_revise', action='store_true')
    parser.add_argument('--center_adaptive', action='store_true')
    parser.add_argument('--siren_on_the_logits', action='store_true')
    parser.add_argument('--dismax', action='store_true')
    parser.add_argument('--dismax_weight', type=float, default=1)
    parser.add_argument('--gpu_option', default=0, type=int)

    return parser
