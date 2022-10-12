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
from sklearn import covariance
import faiss


recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--normalize', default=1, type=int)
parser.add_argument('--ow', default=0, type=int)
parser.add_argument('--pen', default=0, type=int)
parser.add_argument('--open', default=0, type=int)
parser.add_argument('--pro', default=0, type=int)
parser.add_argument('--pro_length', default=128, type=int)
parser.add_argument('--gpu_option', default=0, type=int)
args = parser.parse_args()

# name = 'supervised_fine_tune_full_pascal'
name = args.name
if args.ow:
    length = 21
elif args.pen:
    length = 256
elif args.pro:
    length = args.pro_length
else:
    length = 20
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def print_results(results, stypes):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']

    for stype in stypes:
        print(' OOD detection method: ' + stype)
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('\n{val:6.2f}'.format(val=100.*results[stype]['FPR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['DTERR']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUROC']), end='')
        print(' {val:6.2f}'.format(val=100.*results[stype]['AUIN']), end='')
        print(' {val:6.2f}\n'.format(val=100.*results[stype]['AUOUT']), end='')
        print('')

# ID data
if args.open == 0:
    if args.gpu_option == 2:
        if args.pen:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name +'/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length+1)
            id_train_data = id_train_data[:,:-1].numpy()

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        elif args.pro:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        else:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name +'/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length).numpy()
            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
    elif args.gpu_option == 1:
        if args.pen:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        elif args.pro:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        else:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length).numpy()
            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
    else:
        if args.pen:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        elif args.pro:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        else:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length).numpy()
            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
else:
    if args.gpu_option == 2:
        if args.pen:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name +'/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length+1)
            id_train_data = id_train_data[:,:-1].numpy()

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-open-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        elif args.pro:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-open-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        else:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name +'/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length).numpy()
            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-open-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
    elif args.gpu_option == 1:
        if args.pen:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        elif args.pro:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        else:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length).numpy()
            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
    else:
        if args.pen:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        elif args.pro:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            id_train_data = id_train_data[:, :-1].numpy()

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()
        else:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length).numpy()
            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length).numpy()

            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length).numpy()


normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))
# prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x[:, indices]) for indices in (range(0, 24), range(24, 132), range(132, 282), range(282, 624))], axis=1))
# # breakpoint()
if args.normalize:
    id_train_data = prepos_feat(id_train_data)
    all_data_in = prepos_feat(all_data_in)
    all_data_out = prepos_feat(all_data_out)
else:
    id_train_data = np.ascontiguousarray(id_train_data)

index = faiss.IndexFlatL2(id_train_data.shape[1])
index.add(id_train_data)

for K in [1,10,20,50,100,200,500,1000,3000,5000]:
# for K in [10]:
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


print(len(scores_in))
print(len(scores_ood_test))









