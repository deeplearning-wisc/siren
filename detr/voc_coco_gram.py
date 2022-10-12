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
# id_train_data = id_train_data[filter].numpy()
# labels = labels[filter].numpy()


def G_p(ob, p):
    temp = ob.detach()

    temp = temp ** p
    # breakpoint()
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = ((torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    return temp


mins = []
maxs = []
id_data_train = id_train_data
id_data = all_data_in
ood_data = all_data_out
power = range(1, 11)
for index in range(20):
    mins.append([])
    maxs.append([])
    for index1 in range(10):
        mins[index].append([])
        maxs[index].append([])
# breakpoint()
for index in range(20):
    # breakpoint()
    data_class = (labels == index).nonzero().view(-1)
    batch = id_data_train[data_class]#[0:500]
    for p, P in enumerate(power):
        g_p = G_p(batch, P)
        # breakpoint()
        current_min = g_p.min(dim=0, keepdim=True)[0]
        current_max = g_p.max(dim=0, keepdim=True)[0]
        # breakpoint()
        mins[index][p] = current_min
        maxs[index][p] = current_max

batch_deviations = []
test_confs_PRED = []

labels_val_in = torch.from_numpy(np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/pascal/id-logits.npy'))
labels_val_in = torch.sigmoid(labels_val_in).max(-1)[1]


labels_ood = torch.from_numpy(np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/pascal/ood-logits.npy'))
labels_ood = torch.sigmoid(labels_ood).max(-1)[1]

for index in range(20):
    data_class = (labels_val_in == index).nonzero().view(-1)
    batch = id_data[data_class]#[0:500]
    test_confs_PRED.append(batch[:, index].cpu().data.numpy())

    dev = 0
    for p, P in enumerate(power):
        g_p = G_p(batch, P)

        dev += (F.relu(mins[index][p] - g_p) / torch.abs(mins[index][p] + 10 ** -6)).sum(dim=1, keepdim=True)
        dev += (F.relu(g_p - maxs[index][p]) / torch.abs(maxs[index][p] + 10 ** -6)).sum(dim=1, keepdim=True)
    batch_deviations.append(dev.cpu().detach().numpy())

test_confs_PRED = np.concatenate(test_confs_PRED, 0)
id_deviations = np.concatenate(batch_deviations, axis=0) / test_confs_PRED[:, np.newaxis]


batch_deviations = []
test_confs_PRED = []

for index in range(20):
    data_class = (labels_ood == index).nonzero().view(-1)
    batch = ood_data[data_class]#[0:500]
    test_confs_PRED.append(batch[:, index].cpu().data.numpy())

    dev = 0
    for p, P in enumerate(power):
        g_p = G_p(batch, P)

        dev += (F.relu(mins[index][p] - g_p) / torch.abs(mins[index][p] + 10 ** -6)).sum(dim=1, keepdim=True)
        dev += (F.relu(g_p - maxs[index][p]) / torch.abs(maxs[index][p] + 10 ** -6)).sum(dim=1, keepdim=True)
    batch_deviations.append(dev.cpu().detach().numpy())

test_confs_PRED = np.concatenate(test_confs_PRED, 0)
ood_deviations = np.concatenate(batch_deviations, axis=0) / test_confs_PRED[:, np.newaxis]
id_score = id_deviations.reshape(-1)
ood_score = ood_deviations.reshape(-1)


print(len(id_score))
print(len(ood_score))

measures = get_measures(-id_score, -ood_score, plot=False)

# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'energy')

