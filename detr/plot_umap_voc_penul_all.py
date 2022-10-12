import umap
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import argparse


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--logits', default=0, type=int)
parser.add_argument('--pro', default=0, type=int)
parser.add_argument('--kappa', default=0, type=int)
parser.add_argument('--pro_length', default=0, type=int)
parser.add_argument('--name', default=1., type=str)
args = parser.parse_args()

sns.set(context="paper", style="white")
sns.set_style("dark")
class_dict = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
new_class_dict = {}
for index in range(20):
    new_class_dict[index] = class_dict[index]



if args.logits:
    data = np.load('/nobackup-slow/my_xfdu/detr_out/exps/' + args.name + '/id-logits_maha_train.npy')#.item()
    data_pen = np.load('/nobackup-slow/my_xfdu/detr_out/exps/' + args.name + '/id-pen_maha_train.npy')#.item()
    shape = 20
elif args.pro:
    data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-pro_maha_train.npy')  # .item()
    shape = args.pro_length
else:
    data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/id-pen_maha_train.npy')  # .item()
    shape = 256



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
data = data_dict
data[14] = data_dict[14][:2000]
data_dict[14] = data_dict[14][:2000]

for i in range(20):
    if i == 0:
        data_preprocess = data[i]
    else:
        data_preprocess = np.concatenate((data_preprocess, data[i]), 0)

data_preprocess = np.array(data_preprocess).reshape(-1, shape)
targets = []
for i in range(20):
    for _ in range(len(data_dict[i])):
        targets.append(new_class_dict[i])
targets = np.array(targets).reshape(-1)

data_preprocess = F.normalize(torch.from_numpy(data_preprocess), p=2,dim=-1).numpy()
# breakpoint()
reducer = umap.UMAP(random_state=43,n_neighbors=15, min_dist=0.3, n_components=2, metric='euclidean')#30, 0.6
embedding = reducer.fit_transform(data_preprocess)


fig, ax = plt.subplots(figsize=(12, 12))
# color = mnist.target.astype(int)
def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


classes = [str(hhh) for hhh in range(20)]
# color = targets.astype(int)#[index for index in range(20)]#
color = get_cmap(20)
# color = plt.cm.coolwarm(np.linspace(0.1,0.9,11))
# selected = np.random.choice(20, 10, replace=False)
index = 0
sum = 0
if args.kappa:
    kappa = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
    kappa = kappa.reshape(-1)
for i in range(0, 20):
    plt.scatter(embedding[:, 0][sum: sum + len(data_dict[i])],
                embedding[:, 1][sum: sum + len(data_dict[i])],
                # c=color(i),
                label=new_class_dict[index], cmap="autumn", s=10, marker="^")
    if args.kappa:
        plt.text(embedding[:, 0][sum: sum + len(data_dict[i])][-1], embedding[:, 1][sum: sum + len(data_dict[i])][-1],
                 new_class_dict[i] + '_' + str(kappa[i]))
    sum += len(data_dict[i])
    index += 1

# plt.legend(fontsize=20)
# ax.legend(loc='lower left',markerscale=9)
plt.setp(ax, xticks=[], yticks=[])
plt.savefig('./voc_coco_umap_ours_pen1_new111.jpg', dpi=250)
# plt.show()