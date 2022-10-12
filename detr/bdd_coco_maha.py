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
parser.add_argument('--pro', default=0, type=int)
parser.add_argument('--open', default=0, type=int)
parser.add_argument('--normalize', default=0, type=int)
parser.add_argument('--pro_length', default=128, type=int)
parser.add_argument('--gpu_option', default=0, type=int)
args = parser.parse_args()

# name = 'supervised_fine_tune_full_pascal'
name = args.name
if args.ow:
    length = 11
elif args.pen:
    length = 256
elif args.pro:
    length = args.pro_length
else:
    length = 10
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
            all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
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
# id_train_data = id_train_data[filter].numpy()
# labels = labels[filter].numpy()




all_data_in = all_data_in[:100000]


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
        mean_class = np.zeros((10, 256))
    elif index == 1:
        data_train = id_train_data[:, 256: 256 + 1024]
        mean_class = np.zeros((10, 1024))
    elif index == 2:
        data_train = id_train_data[:, 1024+256:2048+256]
        mean_class = np.zeros((10, 1024))
    else:
        # breakpoint()
        if args.normalize == 0:
            data_train = id_train_data[:, 0:length]
        else:
            data_train = id_train_data[:, 0:length] / np.linalg.norm(id_train_data[:, 0:length], ord=2, axis=-1,
                                                                 keepdims=True)
        mean_class = np.zeros((10, length))
    # mean_class = np.zeros((20, 21))
    # class_id = id_train_data[:, -1].reshape(-1,1)
    class_id = labels.reshape(-1,1)
    # breakpoint()
    data_train = np.concatenate([data_train, class_id], 1)
    sample_dict = {}

    for i in range(10):
        sample_dict[i] = []
    for data in data_train:
        # print(data.shape)
        # breakpoint()
        if int(data[-1]) == 10:
            print('hhh')
            continue
        # if len(sample_dict[int(data[-1])]) <= 1:
        mean_class[int(data[-1])] += data[:-1]
        sample_dict[int(data[-1])].append(data[:-1])
        # else:
        #     continue
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
        # pass
        pass
        # mean_class[5] = np.random.normal(0, 1, 21)
        # sample_dict[5] = [np.random.normal(0, 1, 21)]
    for i in range(10):
        mean_class[i] = mean_class[i] / len(sample_dict[i])
        for data in sample_dict[i]:
            data -= mean_class[i]
    # breakpoint()
    mean_class = torch.from_numpy(mean_class)
    group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
    X = 0

    for i in range(10):
        if i == 0:
            X = sample_dict[i]
        else:
            if i == 5:
                continue
            else:
                X = np.concatenate([X, sample_dict[i]], 0)
    # breakpoint()
    # group_lasso.fit(X)
    # temp_precision = group_lasso.precision_

    X = group_lasso._validate_data(X)
    group_lasso.location_ = X.mean(0)
    # covariance = empirical_covariance(X, assume_centered=self.assume_centered)
    
    X = np.asarray(X)

    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array"
        )
    covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    import scipy
    from sklearn.utils.validation import check_array

    covariance = check_array(covariance)
    group_lasso.covariance_ = covariance
    # breakpoint()
    temp_precision = np.linalg.pinv(covariance)



    temp_precision = torch.from_numpy(temp_precision).float()
    precision = temp_precision
    covariance_list.append(precision)
    mean_list.append(mean_class)

print(len(all_data_out))
print(len(all_data_in))




gaussian_score = 0
for i in range(10):
    # breakpoint()
    if i != 5:
        batch_sample_mean = mean_list[-1][i]
        # breakpoint()
        if args.normalize == 0:
            zero_f = all_data_in - batch_sample_mean
        else:
            zero_f = F.normalize(all_data_in, p=2, dim=-1)-batch_sample_mean
        # breakpoint()
        term_gau = -0.5*torch.mm(torch.mm(zero_f.float(), covariance_list[-1].float()), zero_f.float().t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1,1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        # breakpoint()
id_score, _ = torch.max(gaussian_score, dim=1)

# for ID data.
gaussian_score = 0
for i in range(10):
    if i != 5:
        batch_sample_mean = mean_list[-1][i]
        if args.normalize == 0:
            zero_f = all_data_out - batch_sample_mean
        else:
            zero_f = F.normalize(all_data_out, p=2, dim=-1)-batch_sample_mean

        term_gau = -0.5*torch.mm(torch.mm(zero_f.float(), covariance_list[-1].float()), zero_f.float().t()).diag()
        if i == 0:
            gaussian_score = term_gau.view(-1,1)
        else:
            gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
ood_score, _ = torch.max(gaussian_score, dim=1)

print(len(id_score))
print(len(ood_score))

measures = get_measures(id_score.cpu().data.numpy(), ood_score.cpu().data.numpy(), plot=False)

# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'energy')




