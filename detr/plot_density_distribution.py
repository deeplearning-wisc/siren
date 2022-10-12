import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from vMF import *

parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--logits', default=0, type=int)
parser.add_argument('--pro', default=0, type=int)
parser.add_argument('--pro_length', default=0, type=int)
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--sample', default=10000, type=int)
parser.add_argument('--select', default=1, type=int)
args = parser.parse_args()

sns.set(context="paper", style="white")



import torch
import torch.nn.functional as F
name = 'pascal_center_project_dim_16_weight_1.5_t_0.1'
data_preprocess1 = torch.from_numpy(np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/'+name+'/id-pro_maha_train.npy'))
# data_preprocess1 = F.normalize(data_preprocess1, p=2,dim=-1)



if args.logits:
    shape = 20
elif args.pro:
    shape = args.pro_length
else:
    shape = 256


data = data_preprocess1
data_dict = {}
for index in range(len(data)):
    sub = data[index]
    if not args.logits:
        label = int(sub[-1])
    else:
        label = int(data_pen[index][-1])
    if label in list(data_dict.keys()):
        if not args.logits:
            data_dict[label] = np.concatenate((data_dict[label], sub[:-1].reshape(-1, shape)), 0)
        else:
            data_dict[label] = np.concatenate((data_dict[label], sub.reshape(-1, shape)), 0)
    else:
        if not args.logits:
            data_dict[label] = sub[:-1].reshape(-1,shape)
        else:
            data_dict[label] = sub.reshape(-1, shape)


# breakpoint()
data = F.normalize(torch.from_numpy(data_dict[2]), p=2,dim=-1)#[:1000]
data_mean = data.mean(0)
xm_norm = (data_mean ** 2).sum().sqrt()
mu = data_mean / xm_norm
kappa = (len(data_mean) * xm_norm - xm_norm ** 3) / (1 - xm_norm ** 2)
# breakpoint()
data_preprocess2 = None
density_ood = []
for index in range(1000):
    ood_samples = sample_vMF(mu.numpy(), kappa.numpy(), args.sample)
    density1 = density(mu.numpy(), kappa.numpy(), ood_samples)
    if data_preprocess2 == None:
        data_preprocess2 = torch.from_numpy(
            ood_samples[(-density1).topk(args.select)[1]]).float().view(args.select,-1)
    else:
        data_preprocess2 = torch.cat([data_preprocess2, torch.from_numpy(
            ood_samples[(-density1).topk(args.select)[1]]).float().view(args.select,-1)], 0)
    density_ood.append((-density1).topk(args.select)[0].numpy())
    # print(-(-density1).topk(args.select)[0])
import pandas as pd
import seaborn as sns
# breakpoint()
plt.figure(figsize=(5.5,3))
# plot of 2 variables
id_pd = pd.Series(density(mu.numpy(), kappa.numpy(), data.numpy()).numpy())
breakpoint()
ood_pd = pd.Series(-np.stack(density_ood).reshape(-1))
p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='OOD')
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('outlier_density.jpg', dpi=250)


