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

recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--ow', default=0, type=int)
parser.add_argument('--pen', default=0, type=int)
parser.add_argument('--open', default=0, type=int)
parser.add_argument('--pro', default=0, type=int)
parser.add_argument('--use_es', default=0, type=int)
parser.add_argument('--class_dependent', default=0, type=int)
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


# ID data

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))



# filter = id_train_data.sigmoid()[:, :-1].max(1)[0]> 0.5
if args.open == 0:
    if args.gpu_option == 2:
        if args.pen:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)

        elif args.pro:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
        else:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length)
            labels = id_train_data.sigmoid().max(1)[1]


            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
            # breakpoint()
    elif args.gpu_option == 1:
        if args.pen:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
            # breakpoint()

        elif args.pro:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
        else:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length)
            labels = id_train_data.sigmoid().max(1)[1]

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
    else:
        if args.pen:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)

        elif args.pro:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
        else:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length)
            labels = id_train_data.sigmoid().max(1)[1]

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)

else:
    if args.gpu_option == 2:
        if args.pen:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-open-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)

        elif args.pro:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-open-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
        else:
            id_train_data = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length)
            labels = id_train_data.sigmoid().max(1)[1]


            all_data_in = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/my_xfdu/detr_out/exps/' + name + '/ood-open-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
            # breakpoint()
    elif args.gpu_option == 1:
        if args.pen:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
            # breakpoint()

        elif args.pro:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
        else:
            id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length)
            labels = id_train_data.sigmoid().max(1)[1]

            all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
    else:
        if args.pen:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pen.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)

        elif args.pro:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
            labels = id_train_data[:, -1].int()
            id_train_data = id_train_data[:, :-1]

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pro.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
        else:
            id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits_maha_train.npy')
            id_train_data = torch.from_numpy(id_train_data).reshape(-1, length)
            labels = id_train_data.sigmoid().max(1)[1]

            all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-logits.npy')
            all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-logits.npy')
            all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)

if args.use_es:
    if args.gpu_option == 1:
        mean_load = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/proto.npy')
        kappa_load = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
        print(kappa_load)
    elif args.gpu_option == 0:
        mean_load = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/proto.npy')
        kappa_load = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
        print(kappa_load)
    else:
        mean_load = np.load('/nobackup/my_xfdu/detr_out/exps/' + args.name + '/proto.npy')
        kappa_load = np.load('/nobackup/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
        print(kappa_load)

data_train = id_train_data[:, 0:length] / np.linalg.norm(id_train_data[:, 0:length], ord=2, axis=-1, keepdims=True)
center_metric = []
for index in range(20):

    if args.use_es:
        center_metric.append(
            (data_train[labels == index]* mean_load[index].view(1,-1)).sum(-1).mean())

    else:
        proto = data_train[labels == index].mean(0)
        proto = F.normalize(proto,p=2,dim=-1)
        # breakpoint()
        tmp = (data_train[labels == index] *  proto.view(1,-1)).sum(-1)
        center_metric.append(tmp.mean())
print(torch.stack(center_metric).mean())
# breakpoint()