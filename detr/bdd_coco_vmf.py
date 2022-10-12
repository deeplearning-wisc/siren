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
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--use_trained_params', default=1, type=int)
parser.add_argument('--pro_length', default=64, type=int)
args = parser.parse_args()


name = args.name
if args.name == 'siren':
    length = args.pro_length
else:
    length = 256

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


# ID data

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))








if args.name == 'siren':
    id_train_data = np.load('./snapshots/bdd/' + name + '/id-pro_maha_train.npy')
    id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
    labels = id_train_data[:, -1].int()
    id_train_data = id_train_data[:, :-1]
    all_data_in = np.load('./snapshots/bdd/' + name + '/id-pro.npy')
    all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
    all_data_out = np.load('./snapshots/bdd/' + name + '/ood-pro.npy')
    all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
else: # for vanilla
    id_train_data = np.load('./snapshots/bdd/' + name + '/id-pen_maha_train.npy')
    id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
    labels = id_train_data[:, -1].int()
    id_train_data = id_train_data[:, :-1]
    all_data_in = np.load('./snapshots/bdd/' + name + '/id-pen.npy')
    all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
    all_data_out = np.load('./snapshots/bdd/' + name + '/ood-pen.npy')
    all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)





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
        data_train = id_train_data[:, 0:length] / np.linalg.norm(id_train_data[:, 0:length],
                                                                 ord=2, axis=-1, keepdims=True)
        mean_class = np.zeros((10, length))
    class_id = labels.reshape(-1,1)
    data_train = np.concatenate([data_train, class_id], 1)
    sample_dict = {}

    for i in range(10):
        sample_dict[i] = []
    for data in data_train:
        if int(data[-1]) == 10:
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
    for i in range(10):
        mean_class[i] = mean_class[i] / len(sample_dict[i])
    mean_class = torch.from_numpy(mean_class)
    mean_list.append(mean_class)

print(len(all_data_out))
print(len(all_data_in))

from vMF import density

if args.use_trained_params:
    mean_load = np.load('./snapshots/bdd/' + args.name + '/proto.npy')
    kappa_load = np.load('./snapshots/bdd/' + args.name + '/kappa.npy')
    print(kappa_load)

gaussian_score = 0
gaussian_score1 = 0
for i in range(10):
    if args.use_trained_params == 0:
        batch_sample_mean = mean_list[-1][i]
        xm_norm = (batch_sample_mean ** 2).sum().sqrt()
        mu0 = batch_sample_mean / xm_norm
        print(xm_norm)
        kappa0 = (len(batch_sample_mean) * xm_norm - xm_norm ** 3) / (1 - xm_norm ** 2)

        prob_density = density(mu0.numpy(), kappa0.numpy(), F.normalize(all_data_in, p=2, dim=-1).numpy())
        prob_density1 = density(mu0.numpy(), kappa0.numpy(), F.normalize(all_data_out, p=2, dim=-1).numpy())
        print(kappa0)
    else:
        mu0 = mean_load[i]
        kappa0 = kappa_load[0][i]

        prob_density = (density(mu0, kappa0, F.normalize(all_data_in, p=2, dim=-1).numpy()))
        prob_density1 = (density(mu0, kappa0, F.normalize(all_data_out, p=2, dim=-1).numpy()))


    # breakpoint()
    if i == 0:
        gaussian_score = prob_density.view(-1,1)
        gaussian_score1 = prob_density1.view(-1,1)
    else:
        gaussian_score = torch.cat((gaussian_score, prob_density.view(-1,1)), 1)
        gaussian_score1 = torch.cat((gaussian_score1, prob_density1.view(-1, 1)), 1)





id_score, _ = torch.max(gaussian_score, dim=1)
ood_score, _ = torch.max(gaussian_score1, dim=1)


print(len(id_score))
print(len(ood_score))

# vMF score
measures = get_measures(id_score.cpu().data.numpy(), ood_score.cpu().data.numpy(), plot=False)
print_measures(measures[0], measures[1], measures[2], 'energy')



# knn score
id_train_data = prepos_feat(id_train_data)
all_data_in = prepos_feat(all_data_in)
all_data_out = prepos_feat(all_data_out)


import faiss
index = faiss.IndexFlatL2(id_train_data.shape[1])
index.add(id_train_data)
index.add(id_train_data)
for K in [1, 5, 10 ,20, 50, 100, 200, 300, 400, 500]:
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




