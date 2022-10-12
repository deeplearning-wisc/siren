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
if args.open==0:
    if args.gpu_option == 2:
        data_dir_id = '/nobackup/my_xfdu/detr_out/exps/' + args.name + '/id-sampling.npy'
        data_dir_ood = '/nobackup/my_xfdu/detr_out/exps/' + args.name + '/ood-sampling.npy'
    elif args.gpu_option == 1:
        data_dir_id = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-sampling.npy'
        data_dir_ood = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-sampling.npy'
    else:
        data_dir_id = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-sampling.npy'
        data_dir_ood = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-sampling.npy'
else:
    if args.gpu_option == 2:
        data_dir_id = '/nobackup/my_xfdu/detr_out/exps/'+args.name+'/id-sampling.npy'
        data_dir_ood = '/nobackup/my_xfdu/detr_out/exps/'+args.name+'/ood-open-sampling.npy'
    elif args.gpu_option == 1:
        data_dir_id = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-sampling.npy'
        data_dir_ood = '/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-open-sampling.npy'
    else:
        data_dir_id = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-sampling.npy'
        data_dir_ood = '/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/ood-open-sampling.npy'
# ID data
id_data = np.load(data_dir_id)
ood_data = np.load(data_dir_ood)

id = 0
T = 1
id_score = []
ood_score = []
# breakpoint()

id_score = id_data.min(-1)
ood_score = ood_data.min(-1)

measures = get_measures(id_score, ood_score, plot=False)
print_measures(measures[0], measures[1], measures[2], 'energy')
# breakpoint()

print(len(id_score))
print(len(ood_score))



for index in range(20):
    plt.figure(figsize=(5.5,3))
    # plot of 2 variables
    id_pd = pd.Series(id_data[:,index])
    ood_pd = pd.Series(ood_data[:,index])
    p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
    p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='OOD')
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('./mc_relu/voc_coco_mc_relu_' +str(index) + '.jpg', dpi=250)
    plt.clf()


