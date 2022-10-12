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
from sklearn import svm

recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--ow', default=0, type=int)
parser.add_argument('--pen', default=0, type=int)
parser.add_argument('--pro', default=1, type=int)
parser.add_argument('--pro_label', default=0, type=int)
parser.add_argument('--pro_nu', default=0.99, type=float)
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
id_train_data = id_train_data.numpy()



# clf = svm.OneClassSVM(kernel="rbf", gamma='scale', nu=args.pro_nu)
# selected = labels.numpy() == args.pro_label
# id_train_data1 = id_train_data[selected]
# clf.fit(id_train_data1)
# # breakpoint()
# id_pro = clf.decision_function(all_data_in.numpy())
# ood_pro = clf.decision_function(all_data_out.numpy())
# measures = get_measures(-id_pro, -ood_pro, plot=False)
#
# print_measures(measures[0], measures[1], measures[2], 'ova')
# breakpoint()
print(len(all_data_in))
print(len(all_data_out))


for label in range(20):
    clf = svm.OneClassSVM(nu=args.pro_nu, kernel='rbf',  gamma='scale')
    selected = labels.numpy() == label #args.pro_label
    id_train_data1 = id_train_data[selected]
    clf.fit(id_train_data1)
    # breakpoint()
    id_pro = clf.decision_function(all_data_in.numpy())
    ood_pro = clf.decision_function(all_data_out.numpy())
    measures = get_measures(-id_pro, -ood_pro, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'ova')

    if label == 0:
        id_score = -torch.from_numpy(id_pro).view(-1,1)
        ood_score = -torch.from_numpy(ood_pro).view(-1, 1)
    else:
        id_score = torch.cat((id_score,
                                    -torch.from_numpy(id_pro).view(-1,1)), 1)
        ood_score = torch.cat((ood_score,
                              -torch.from_numpy(ood_pro).view(-1, 1)), 1)


id_score1, _ = torch.max(id_score, dim=1)
ood_score1, _ = torch.max(ood_score, dim=1)
measures = get_measures(id_score1.numpy(), ood_score1.numpy(), plot=False)
# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'ova max')

id_score1, _ = torch.min(id_score, dim=1)
ood_score1, _ = torch.min(ood_score, dim=1)
measures = get_measures(id_score1.numpy(), ood_score1.numpy(), plot=False)
# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'ova min')

id_score1 = torch.mean(id_score, dim=1)
ood_score1 = torch.mean(ood_score, dim=1)
measures = get_measures(id_score1.numpy(), ood_score1.numpy(), plot=False)
# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'ova mean')
breakpoint()
