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


#parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
#                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('--name', default=1., type=str)
#parser.add_argument('--normalize', default=0, type=int)
#args = parser.parse_args()
#
## name = 'supervised_fine_tune_full_pascal'
#name = args.name

parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=0.5275, type=float)
parser.add_argument('--name', default='vosV3CenterLoss', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
parser.add_argument('--normalize', default=0, type=int)
args = parser.parse_args()


length = 16 
# the projected features for id training data., the last dimension is the label for the feature.
# id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
# id_train_data = np.load('./detection/data/VOC-Detection/faster-rcnn/vosV3CenterLoss/random_seed_0/inference/voc_custom_train/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_0.5275.pkl')['binary_cls']
id_train_data = pickle.load(open('./data/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_train/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
labels = id_train_data['predicted_cls_id']
id_train_data = torch.stack(id_train_data['binary_cls'])
# id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
# labels = id_train_data[:, -1].int()
# id_train_data = id_train_data[:, :-1]


# the projected features for id validation/test data.
#all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
#all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
id_val_data = pickle.load(open('./data/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
labels_val = id_val_data['predicted_cls_id']
id_val_data = torch.stack(id_val_data['binary_cls']).cpu()

## the projected features for ood data.
ood_val_data = pickle.load(open('./data/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/coco_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
labels_ood = ood_val_data['predicted_cls_id']
ood_val_data = torch.stack(ood_val_data['binary_cls']).cpu()
# all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
# all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)

# import ipdb; ipdb.set_trace()


id = 0
T = 1
id_score = []
ood_score = []

mean_list = []
covariance_list = []

if args.normalize == 0:
    data_train = id_train_data
else:
    data_train = id_train_data / np.linalg.norm(id_train_data, ord=2, axis=-1,
                                                         keepdims=True)
mean_class = np.zeros((20, length))
class_id = labels # labels.reshape(-1,1)
data_train = data_train.cpu().numpy()
class_id = class_id.cpu().numpy()
# data_train = np.concatenate([data_train, class_id], 1)
data_train = np.hstack((data_train, np.expand_dims(class_id, 1)))
sample_dict = {}

for i in range(20):
    sample_dict[i] = []
for data in data_train:
    mean_class[int(data[-1])] += data[:-1]
    sample_dict[int(data[-1])].append(data[:-1])

# get the mean of the features.
for i in range(20):
    mean_class[i] = mean_class[i] / len(sample_dict[i])
    # make the features centered.
    for data in sample_dict[i]:
        data -= mean_class[i]
mean_class = torch.from_numpy(mean_class)
# covariance.
group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
X = 0

for i in range(20):
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


#print(len(all_data_out))
#print(len(all_data_in))
print(len(id_val_data))
print(len(ood_val_data))



gaussian_score = 0
for i in range(20):
    # breakpoint()
    batch_sample_mean = mean_list[-1][i]
    # breakpoint()
    if args.normalize == 0:
        # zero_f = all_data_in - batch_sample_mean
        zero_f = id_val_data - batch_sample_mean
    else:
        # zero_f = F.normalize(all_data_in, p=2, dim=-1)-batch_sample_mean
        zero_f = F.normalize(id_val_data, p=2, dim=-1)-batch_sample_mean
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
for i in range(20):
    batch_sample_mean = mean_list[-1][i]
    if args.normalize == 0:
        # zero_f = all_data_out - batch_sample_mean
        zero_f = ood_val_data - batch_sample_mean
    else:
        # zero_f = F.normalize(all_data_out, p=2, dim=-1)-batch_sample_mean
        zero_f = F.normalize(ood_val_data, p=2, dim=-1)-batch_sample_mean

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
print_measures(measures[0], measures[1], measures[2], 'maha')




