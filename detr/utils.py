import os
import torch
import numpy as np
import sklearn
from sklearn import covariance
realmin = 1e-10


def norm(input, p=2, dim=0, eps=1e-12):
    return input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def maha_train_estimation(train_embed, labels_train, length = 16):
    mean_list = []
    covariance_list = []


    for index in range(3, 4):
        if index == 0:
            data_train = id_train_data[:, :256]
            mean_class = np.zeros((20, 256))
        elif index == 1:
            data_train = id_train_data[:, 256: 256 + 1024]
            mean_class = np.zeros((10, 1024))
        elif index == 2:
            data_train = id_train_data[:, 1024 + 256:2048 + 256]
            mean_class = np.zeros((10, 1024))
        else:
            # breakpoint()
            train_embed = train_embed[:, 0:length]
            mean_class = np.zeros((3, length))
        # mean_class = np.zeros((20, 21))
        # class_id = id_train_data[:, -1].reshape(-1,1)
        class_id = labels_train.reshape(-1, 1)
        # breakpoint()
        train_embed = np.concatenate([train_embed, class_id], 1)
        sample_dict = {}

        for i in range(3):
            sample_dict[i] = []
        for data in train_embed:
            mean_class[int(data[-1])] += data[:-1]
            sample_dict[int(data[-1])].append(data[:-1])

        for i in range(3):
            mean_class[i] = mean_class[i] / len(sample_dict[i])
            for data in sample_dict[i]:
                data -= mean_class[i]
        mean_class = torch.from_numpy(mean_class)
        group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
        X = 0

        for i in range(3):
            if i == 0:
                X = sample_dict[i]
            else:
                X = np.concatenate([X, sample_dict[i]], 0)
        group_lasso.fit(X)
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float()
        precision = temp_precision
        covariance_list.append(precision)
        mean_list.append(mean_class)
        return mean_list, covariance_list