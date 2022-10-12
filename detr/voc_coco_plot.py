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
        data_dir_id = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-pen.npy'
        data_dir_ood = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-pen.npy'
    else:
        data_dir_id = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-logits.npy'
        data_dir_ood = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-logits.npy'
else:
    if args.gpu_option == 2:
        data_dir_id = '/nobackup/my_xfdu/detr_out/exps/' + args.name + '/id-logits.npy'
        data_dir_ood = '/nobackup/my_xfdu/detr_out/exps/' + args.name + '/ood-open-logits.npy'
    elif args.gpu_option == 1:
        data_dir_id = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-pen.npy'
        data_dir_ood = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-open-pen.npy'
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


# def get_dismax_score(logits):
#     probabilities = torch.nn.Softmax(dim=1)(logits)
#     scores = logits.max(dim=1)[0] + logits.mean(dim=1) + (probabilities * torch.log(probabilities)).sum(dim=1)
#     return scores
#
#
# id_score = get_dismax_score(torch.from_numpy(id_data).cuda())
# # breakpoint()
# ood_score = get_dismax_score(torch.from_numpy(ood_data).cuda())
#
# measures = get_measures(id_score.cpu().data.numpy(), ood_score.cpu().data.numpy(), plot=False)
#
#
#
# # if args.energy:
# print_measures(measures[0], measures[1], measures[2], 'energy')
# breakpoint()



# assert len(id_data['inter_feat'][0]) == 21# + 1024
# breakpoint()
# if 'ce' in data_dir_id:
#     id_data_filter = F.softmax(torch.from_numpy(id_data).view(-1,21), 1)
#     ood_data_filter = F.softmax(torch.from_numpy(ood_data).view(-1,21), 1)
#     filter_index_id = id_data_filter[:, :-1].max(1)[0] > 0.5
#     filter_index_ood = ood_data_filter[:, :-1].max(1)[0] > 0.5
# else:
#     id_data_filter = torch.from_numpy(id_data).view(-1,21).sigmoid()
#     ood_data_filter = torch.from_numpy(ood_data).view(-1,21).sigmoid()
#     filter_index_id = id_data_filter[:, :-1].max(1)[0] > 0.5
#     filter_index_ood = ood_data_filter[:, :-1].max(1)[0] > 0.5


# name = 'DETReg_fine_tune_full_pascal_in100'
# data_dir_id = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/'+name+'/id1.npy'
# data_dir_ood = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/'+name+'/ood1.npy'
# # ID data
# id_data = np.load(data_dir_id)
# ood_data = np.load(data_dir_ood)
#
# id_score = -id_data.reshape(-1)[filter_index_id.numpy()]
# ood_score = -ood_data.reshape(-1)[filter_index_ood.numpy()]
#
# breakpoint()
# breakpoint()
# if args.energy:
# id_score = torch.from_numpy(id_data).view(-1,21).sigmoid()[:, -1].numpy()
# ood_score = torch.from_numpy(ood_data).view(-1,21).sigmoid()[:, -1].numpy()
#
#
# id_score = id_score[filter_index_id]
# ood_score = ood_score[filter_index_ood]
# measures = get_measures(-id_score, -ood_score, plot=False)
# print_measures(measures[0], measures[1], measures[2], 'energy')


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

    id_score = torch.from_numpy(id_data).view(-1,21).sigmoid()[:,-1]
    ood_score = torch.from_numpy(ood_data).view(-1, 21).sigmoid()[:,-1]
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'ow-detr')
    ###########
    ########
    print(len(id_score))
    print(len(ood_score))
else:
    id_score = torch.log(1 + torch.exp(torch.from_numpy(id_data).view(-1, 20)))
    id_score = -torch.sum(id_score, 1).numpy()
    ood_score = torch.log(1 + torch.exp(torch.from_numpy(ood_data).view(-1, 20)))
    ood_score = -torch.sum(ood_score, 1).numpy()
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'joint-energy')

    # breakpoint()

    id_score = -np.max(torch.from_numpy(id_data).view(-1, 20).numpy(), axis=1)
    ood_score = -np.max(torch.from_numpy(ood_data).view(-1, 20).numpy(), axis=1)
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'maxlogit')

    # breakpoint()

    id_score = -np.max(F.softmax(torch.from_numpy(id_data).view(-1, 20),1).numpy(), axis=1)
    ood_score = -np.max(F.softmax(torch.from_numpy(ood_data).view(-1, 20),1).numpy(), axis=1)
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'msp')

    temp = 0.1
    # breakpoint()
    id_data = id_data/temp-np.max(id_data/temp, axis=1, keepdims=True)
    ood_data = ood_data / temp - np.max(ood_data/temp, axis=1, keepdims=True)
    id_score = -np.max(F.softmax(torch.from_numpy(id_data).view(-1, 20), 1).numpy(), axis=1)
    ood_score = -np.max(F.softmax(torch.from_numpy(ood_data).view(-1, 20), 1).numpy(), axis=1)
    measures = get_measures(-id_score, -ood_score, plot=False)
    print_measures(measures[0], measures[1], measures[2], 'odin')
    print(len(id_score))
    print(len(ood_score))



