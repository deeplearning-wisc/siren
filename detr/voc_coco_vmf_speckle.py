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
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro-speckle.npy')
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
# # id_train_data = id_train_data[filter].numpy()
# # labels = labels[filter].numpy()

# length = 80
#
# id_train_data = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen_maha_train.npy')
# id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
# labels = id_train_data[:, -1].int()
# id_train_data = id_train_data[:, :-1]
#
# all_data_in = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/id-pen.npy')
# all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
# all_data_out = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + name + '/ood-open-pen.npy')
# all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)



id = 0
T = 1
id_score = []
ood_score = []

mean_list = []
covariance_list = []
# breakpoint()
for index in range(3,4):
    if index == 0:
        data_train = id_train_data[:,:256]
        mean_class = np.zeros((20, 256))
    elif index == 1:
        data_train = id_train_data[:, 256: 256 + 1024]
        mean_class = np.zeros((10, 1024))
    elif index == 2:
        data_train = id_train_data[:, 1024+256:2048+256]
        mean_class = np.zeros((10, 1024))
    else:
        # breakpoint()
        data_train = id_train_data[:, 0:length] / np.linalg.norm(id_train_data[:, 0:length], ord=2, axis=-1, keepdims=True)
        mean_class = np.zeros((20, length))
    class_id = labels.reshape(-1,1)
    data_train = np.concatenate([data_train, class_id], 1)
    sample_dict = {}

    for i in range(20):
        sample_dict[i] = []
    for data in data_train:
        # print(data.shape)
        # breakpoint()
        if int(data[-1]) == 20:
            print('hhh')
            continue

        mean_class[int(data[-1])] += data[:-1]
        sample_dict[int(data[-1])].append(data[:-1])
    if index == 0:
        mean_class[5] = np.random.normal(0, 1, 256)
        sample_dict[5] = [np.random.normal(0, 1, 256)]
    elif index == 1:
        mean_class[5] = np.random.normal(0, 1, 1024)
        sample_dict[5] = [np.random.normal(0, 1, 1024)]
    elif index == 2:
        mean_class[5] = np.random.normal(0, 1, 1024)
        sample_dict[5] = [np.random.normal(0, 1, 1024)]
    elif index == 3:
        pass
    for i in range(20):
        mean_class[i] = mean_class[i] / len(sample_dict[i])
    mean_class = torch.from_numpy(mean_class)
    mean_list.append(mean_class)

print(len(all_data_out))
print(len(all_data_in))

from vMF import density

if args.use_es:
    if args.gpu_option == 1:
        mean_load = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/proto.npy')
        kappa_load = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
        # weight_load = np.load('/nobackup/dataset/my_xfdu/detr_out/exps/' + args.name + '/prior.npy')
        print(kappa_load)
        # print(weight_load)
    elif args.gpu_option == 0:
        mean_load = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/proto.npy')
        kappa_load = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
        print(kappa_load)
    else:
        mean_load = np.load('/nobackup/my_xfdu/detr_out/exps/' + args.name + '/proto.npy')
        kappa_load = np.load('/nobackup/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
        print(kappa_load)

gaussian_score = 0
gaussian_score1 = 0
for i in range(20):
    # breakpoint()
    if args.use_es == 0:
        batch_sample_mean = mean_list[-1][i]
        xm_norm = (batch_sample_mean ** 2).sum().sqrt()
        mu0 = batch_sample_mean / xm_norm
        kappa0 = (len(batch_sample_mean) * xm_norm - xm_norm ** 3) / (1 - xm_norm ** 2)

        prob_density = density(mu0.numpy(), kappa0.numpy(), F.normalize(all_data_in, p=2, dim=-1).numpy())
        prob_density1 = density(mu0.numpy(), kappa0.numpy(), F.normalize(all_data_out, p=2, dim=-1).numpy())
        print(kappa0)
    else:
        mu0 = mean_load[i]
        kappa0 = kappa_load[0][i]
        # prior = weight_load[0][i]
        # prob_density = prior * (density(mu0, kappa0, F.normalize(all_data_in, p=2, dim=-1).numpy()).exp())
        # prob_density1 = prior * (density(mu0, kappa0, F.normalize(all_data_out, p=2, dim=-1).numpy()).exp())

        prob_density = (density(mu0, kappa0, F.normalize(all_data_in, p=2, dim=-1).numpy()))
        prob_density1 = (density(mu0, kappa0, F.normalize(all_data_out, p=2, dim=-1).numpy()))


    # breakpoint()
    if i == 0:
        gaussian_score = prob_density.view(-1,1)
        gaussian_score1 = prob_density1.view(-1,1)
    else:
        gaussian_score = torch.cat((gaussian_score, prob_density.view(-1,1)), 1)
        gaussian_score1 = torch.cat((gaussian_score1, prob_density1.view(-1, 1)), 1)




if args.class_dependent:
    class_weight = torch.from_numpy(np.asarray([
        1285,1208, 1820, 1397, 2116, 909, 4008, 1616, 4338, 1058, 1057,
        2079, 1156, 1141, 15576, 1724, 1347, 1211, 984, 1193
        ])/ 47223)
    CLASSES = (
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    )
    print(class_weight.sum())
    id_score = gaussian_score * class_weight.view(1,-1)
    ood_score = gaussian_score1 * class_weight.view(1,-1)
else:
    id_score, _ = torch.max(gaussian_score, dim=1)
    ood_score, _ = torch.max(gaussian_score1, dim=1)
    # id_score = gaussian_score.sum(-1)
    # ood_score =gaussian_score1.sum(-1)
    # id_score = torch.mean(gaussian_score, dim=1)
    # ood_score = torch.mean(gaussian_score1, dim=1)
    # id_score, _ = torch.min(gaussian_score, dim=1)
    # ood_score, _ = torch.min(gaussian_score1, dim=1)

print(len(id_score))
print(len(ood_score))

measures = get_measures(id_score.cpu().data.numpy(), ood_score.cpu().data.numpy(), plot=False)

# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'energy')


# knn score
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))

if 1:
    id_train_data = prepos_feat(id_train_data)
    all_data_in = prepos_feat(all_data_in)
    all_data_out = prepos_feat(all_data_out)


import faiss

# res = faiss.StandardGpuResources()
#
# index = faiss.GpuIndexFlatL2(res, id_train_data.shape[1])
index = faiss.IndexFlatL2(id_train_data.shape[1])
index.add(id_train_data)
# index = faiss.IndexFlatL2(id_train_data.shape[1])
index.add(id_train_data)
for K in [1, 5, 10 ,20, 50, 100, 200, 300, 400, 500]:
# for K in [300, 500, 600, 700, 800, 900]:
# for K in [10]:
    D, _ = index.search(all_data_in, K)
    scores_in = -D[:,-1]
    all_results = []
    all_score_ood = []
    # for ood_dataset, food in food_all.items():
    D, _ = index.search(all_data_out, K)
    scores_ood_test = -D[:,-1]
    all_score_ood.extend(scores_ood_test)
    results = get_measures(scores_in, scores_ood_test, plot=False)


    print_measures(results[0], results[1], results[2], f'KNN k={K}')
    print()

