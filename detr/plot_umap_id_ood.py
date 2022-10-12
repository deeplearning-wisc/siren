import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--logits', default=0, type=int)
parser.add_argument('--pro', default=0, type=int)
parser.add_argument('--pro_length', default=0, type=int)
parser.add_argument('--name', default=1., type=str)
args = parser.parse_args()

# sns.set(context="paper", style="white")
sns.set_style("dark")


import torch
import torch.nn.functional as F
name = 'pascal_center_project_dim_16_weight_1.5_t_0.1'
data_preprocess1 = torch.from_numpy(np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/'+name+'/id.npy'))
data_preprocess2 = torch.from_numpy(np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/'+name+'/ood.npy'))
# breakpoint()
# data_preprocess1 = data_preprocess1[:,:-1]
data_preprocess1 = F.normalize(data_preprocess1, p=2,dim=-1)
data_preprocess2 = F.normalize(data_preprocess2, p=2,dim=-1)
data_preprocess2 = data_preprocess2.numpy()
data_preprocess1 = data_preprocess1.numpy()
data_preprocess = np.concatenate((data_preprocess1, data_preprocess2), 0)
labels = np.concatenate((np.ones(len(data_preprocess1)), np.zeros(len(data_preprocess2))),-1)
# breakpoint()

shape=16
data_preprocess = np.array(data_preprocess).reshape(-1, shape)
targets = labels



reducer = umap.UMAP(random_state=42,n_neighbors=15, min_dist=0.2, n_components=2, metric='cosine')#30, 0.6
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
                    label=index, cmap="Spectral", s=1)
    else:
        plt.scatter(embedding[:, 0][data_dict[0]:],
                    embedding[:, 1][data_dict[0]:],
                    c='r',
                    label=index, cmap="Spectral", s=1)

plt.legend(fontsize=20)
# ax.legend(markerscale=9)
ax.legend(loc='lower left',markerscale=9)#, bbox_to_anchor=(1, 0.5)
# plt.legend(handles=scatter.legend_elements()[0], labels=classes)
# breakpoint()
plt.setp(ax, xticks=[], yticks=[])
# plt.title("With virtual outliers", fontsize=20)
# plt.savefig('./voc_coco_umap_visual_ours.jpg', dpi=250)
# plt.title("Vanilla detector", fontsize=20)
plt.savefig('./voc_coco_umap_visual_id_ood_cosine.jpg', dpi=250)
# plt.show()