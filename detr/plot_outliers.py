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
ood_samples = sample_vMF(mu.numpy(), kappa.numpy(), args.sample)
density1 = density(mu.numpy(), kappa.numpy(), ood_samples)
# select = 1
data_preprocess2 = torch.from_numpy(ood_samples[(-density1).topk(args.select)[1]]).float().view(args.select,-1)

# breakpoint()

data_preprocess2 = data_preprocess2.numpy()
data_preprocess1 = data.numpy()
data_preprocess = np.concatenate((data_preprocess1, data_preprocess2), 0)
labels = np.concatenate((np.ones(len(data_preprocess1)), np.zeros(len(data_preprocess2))),-1)
# breakpoint()

shape=16
data_preprocess = np.array(data_preprocess).reshape(-1, shape)
targets = labels



reducer = umap.UMAP(random_state=42,n_neighbors=15, min_dist=0.2, n_components=2, metric='euclidean')#30, 0.6
embedding = reducer.fit_transform(data_preprocess)


fig, ax = plt.subplots(figsize=(12, 12))
# color = mnist.target.astype(int)
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


classes = [str(hhh) for hhh in range(2)]
# color = targets.astype(int)#[index for index in range(20)]#
color = get_cmap(2)
# color = plt.cm.coolwarm(np.linspace(0.1,0.9,11))
selected = np.random.choice(20, 10, replace=False)
index = 0
sum = 0
data_dict = [len(data_preprocess1), len(data_preprocess2)]
for i in range(0, 2):
    if i == 0:
        plt.scatter(embedding[:, 0][0: data_dict[0]],
                    embedding[:, 1][0: data_dict[0]],
                    c='b',
                    label=index, cmap="Spectral", s=10)
    else:
        plt.scatter(embedding[:, 0][data_dict[0]:],
                    embedding[:, 1][data_dict[0]:],
                    c='r',
                    label=index, cmap="Spectral", s=100)

# plt.legend(fontsize=20)
# ax.legend(markerscale=9)
# ax.legend(loc='lower left',markerscale=9)#, bbox_to_anchor=(1, 0.5)
# plt.legend(handles=scatter.legend_elements()[0], labels=classes)
# breakpoint()
plt.setp(ax, xticks=[], yticks=[])
# plt.title("With virtual outliers", fontsize=20)
# plt.savefig('./voc_coco_umap_visual_ours.jpg', dpi=250)
# plt.title("Vanilla detector", fontsize=20)
plt.savefig('./voc_coco_outlier' + str(args.sample) + '_' + str(args.select) + '.jpg', dpi=250)
# plt.show()