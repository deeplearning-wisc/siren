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
parser.add_argument('--open', default=0, type=int)
parser.add_argument('--gpu_option', default=0, type=int)
args = parser.parse_args()



concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

# name = 'supervised_fine_tune_full_pascal'
# name = 'DETReg_fine_tune_full_pascal_in'
# name = 'pascal'
if args.open == 0:
    if args.gpu_option == 2:
        data_dir_id = '/nobackup/my_xfdu/detr_out/exps/'+args.name+'/id-logits.npy'
        data_dir_ood = '/nobackup/my_xfdu/detr_out/exps/'+args.name+'/ood-logits.npy'
    elif args.gpu_option == 1:
        data_dir_id = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-logits.npy'
        data_dir_ood = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-logits.npy'
    else:
        data_dir_id = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-logits.npy'
        data_dir_ood = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-logits.npy'
else:
    if args.gpu_option == 2:
        data_dir_id = '/nobackup/my_xfdu/detr_out/exps/' + args.name + '/id-logits.npy'
        data_dir_ood = '/nobackup/my_xfdu/detr_out/exps/' + args.name + '/ood-open-logits.npy'
    elif args.gpu_option == 1:
        data_dir_id = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-logits.npy'
        data_dir_ood = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-open-logits.npy'
    else:
        data_dir_id = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-logits.npy'
        data_dir_ood = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-open-logits.npy'

# ID data
id_data = np.load(data_dir_id)
ood_data = np.load(data_dir_ood)

id = 0
T = 1
id_score = []
ood_score = []


if args.ow:
    id_score = -args.T * torch.logsumexp(torch.from_numpy(id_data).view(-1,21)[:, :-1] / args.T, dim=1).numpy()
    ood_score = -args.T * torch.logsumexp(torch.from_numpy(ood_data).view(-1,21)[:, :-1] / args.T, dim=1).numpy()
    # id_score = id_score[filter_index_id]
    # ood_score = ood_score[filter_index_ood]
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'energy')
    # id_score = -np.max(torch.sigmoid(torch.from_numpy(id_data).view(-1, 21)[:, :-1]).numpy(), 1)
    # ood_score = -np.max(torch.sigmoid(torch.from_numpy(ood_data).view(-1, 21)[:, :-1]).numpy(),1)

    # id_score = -np.max(torch.from_numpy(id_data).view(-1, 21)[:, :-1].numpy(), 1)
    # ood_score = -np.max(torch.from_numpy(ood_data).view(-1, 21)[:, :-1].numpy(), 1)

    id_score = torch.log(1 + torch.exp(torch.from_numpy(id_data).view(-1, 21)[:, :-1]))
    id_score = -torch.max(id_score, 1)[0].numpy()
    ood_score = torch.log(1 + torch.exp(torch.from_numpy(ood_data).view(-1, 21)[:, :-1]))
    ood_score = -torch.max(ood_score, 1)[0].numpy()

    # id_score = id_score[filter_index_id]
    # ood_score = ood_score[filter_index_ood]
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'joint-energy')

    id_score = -np.max(F.softmax(torch.from_numpy(id_data).view(-1,21)[:, :-1], dim=1).numpy(), axis=1)
    ood_score = -np.max(F.softmax(torch.from_numpy(ood_data).view(-1,21)[:, :-1], dim=1).numpy(), axis=1)


    # id_score = id_score[filter_index_id]
    # ood_score = ood_score[filter_index_ood]

    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'msp')
    ###########
    ########
    print(len(id_score))
    print(len(ood_score))
else:
    id_data = id_data[:100000]
    id_score = torch.log(1 + torch.exp(torch.from_numpy(id_data).view(-1, 10)))
    id_score = -torch.sum(id_score, 1).numpy()
    ood_score = torch.log(1 + torch.exp(torch.from_numpy(ood_data).view(-1, 10)))
    ood_score = -torch.sum(ood_score, 1).numpy()
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'joint-energy')

    # breakpoint()

    id_score = -np.max(torch.from_numpy(id_data).view(-1, 10).numpy(), axis=1)
    ood_score = -np.max(torch.from_numpy(ood_data).view(-1, 10).numpy(), axis=1)
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'maxlogit')

    # breakpoint()

    id_score = -np.max(F.softmax(torch.from_numpy(id_data).view(-1, 10),1).numpy(), axis=1)
    ood_score = -np.max(F.softmax(torch.from_numpy(ood_data).view(-1, 10),1).numpy(), axis=1)
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'msp')

    temp = 1000
    # breakpoint()
    id_data = id_data/temp-np.max(id_data/temp, axis=1, keepdims=True)
    ood_data = ood_data / temp - np.max(ood_data/temp, axis=1, keepdims=True)
    id_score = -np.max(F.softmax(torch.from_numpy(id_data).view(-1, 10), 1).numpy(), axis=1)
    ood_score = -np.max(F.softmax(torch.from_numpy(ood_data).view(-1, 10), 1).numpy(), axis=1)
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'odin')
    print(len(id_score))
    print(len(ood_score))



