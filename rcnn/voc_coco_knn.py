import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from metric_utils import *
import sklearn
import mpmath
from sklearn import covariance

recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=0.5275, type=float)
parser.add_argument('--length', default=64, type=int)
parser.add_argument('--name', default='center64_0.1', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
parser.add_argument('--use_es', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--open', default=0, type=int)
args = parser.parse_args()
parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)






name = args.name
if args.gpu == 0:
    prefix = '/nobackup/dataset/'
else:
    prefix = '/nobackup/'

length = args.length

id_train_data = pickle.load(open(prefix + 'my_xfdu/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_train/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
id_val_data = pickle.load(open(prefix + 'my_xfdu/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# labels_val = id_val_data['predicted_cls_id']
id_train_data = torch.stack(id_train_data['projections']).cpu()
id_val_data = torch.stack(id_val_data['projections']).cpu()
if args.open:
    ood_val_data = pickle.load(open(
        prefix + 'my_xfdu/VOC-Detection/' + args.model + '/' + args.name + '/random_seed' + '_' + str(
            args.seed) + '/inference/openimages_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_' + str(
            args.thres) + '.pkl', 'rb'))
else:
    ood_val_data = pickle.load(open(prefix + 'my_xfdu/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/coco_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# labels_ood = ood_val_data['predicted_cls_id']
ood_val_data = torch.stack(ood_val_data['projections']).cpu()


normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))

if 1:
    id_train_data = prepos_feat(id_train_data)
    all_data_in = prepos_feat(id_val_data)
    all_data_out = prepos_feat(ood_val_data)


import faiss
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, id_train_data.shape[1])


index.add(id_train_data)
for K in [1, 5, 10 ,20, 50, 100, 200, 300, 400, 500]:
    D, _ = index.search(all_data_in, K)
    scores_in = -D[:,-1]
    all_results = []
    all_score_ood = []
    # for ood_dataset, food in food_all.items():
    D, _ = index.search(all_data_out, K)
    scores_ood_test = -D[:,-1]
    all_score_ood.extend(scores_ood_test)
    results = get_measures(scores_in, scores_ood_test, plot=False)


    print_measures(results[0], results[1], results[2], f'KNN k={K}')
    print()




