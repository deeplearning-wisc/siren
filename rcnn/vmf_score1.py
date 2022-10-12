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
import mpmath
from sklearn import covariance

recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=0.5275, type=float)
parser.add_argument('--name', default='vosV3CenterLoss', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
parser.add_argument('--use_es', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--open', default=0, type=int)
args = parser.parse_args()
parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)





class vMFLogPartition(torch.autograd.Function):
    '''
    Evaluates log C_d(kappa) for vMF density
    Allows autograd wrt kappa
    '''

    besseli = np.vectorize(mpmath.besseli)
    log = np.vectorize(mpmath.log)
    nhlog2pi = -0.5 * np.log(2 * np.pi)

    @staticmethod
    def forward(ctx, *args):

        '''
        Args:
            args[0] = d; scalar (> 0)
            args[1] = kappa; (> 0) torch tensor of any shape

        Returns:
            logC = log C_d(kappa); torch tensor of the same shape as kappa
        '''

        d = args[0]
        kappa = args[1]

        s = 0.5 * d - 1

        # log I_s(kappa)
        mp_kappa = mpmath.mpf(1.0) * kappa.detach().cpu().numpy()
        mp_logI = vMFLogPartition.log(vMFLogPartition.besseli(s, mp_kappa))
        logI = torch.from_numpy(np.array(mp_logI.tolist(), dtype=float)).to(kappa)

        if (logI != logI).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        logC = d * vMFLogPartition.nhlog2pi + s * kappa.log() - logI

        # save for backard()
        ctx.s, ctx.mp_kappa, ctx.logI = s, mp_kappa, logI

        return logC

    @staticmethod
    def backward(ctx, *grad_output):

        s, mp_kappa, logI = ctx.s, ctx.mp_kappa, ctx.logI

        # log I_{s+1}(kappa)
        mp_logI2 = vMFLogPartition.log(vMFLogPartition.besseli(s + 1, mp_kappa))
        logI2 = torch.from_numpy(np.array(mp_logI2.tolist(), dtype=float)).to(logI)

        if (logI2 != logI2).sum().item() > 0:  # there is nan
            raise ValueError('NaN is detected from the output of log-besseli()')

        dlogC_dkappa = -(logI2 - logI).exp()

        return None, grad_output[0] * dlogC_dkappa



def density(mu, kappa, samples):
    mu = torch.from_numpy(mu)
    kappa = torch.from_numpy(np.asarray(kappa))
    samples = torch.from_numpy(samples)
    dotp = (mu.unsqueeze(0) * samples).sum(1)
    # breakpoint()
    logC = vMFLogPartition.apply(len(mu), kappa.float())
    logliks = kappa * dotp + logC

    return logliks


name = args.name
if args.gpu == 0:
    prefix = '/nobackup-slow/dataset/'
else:
    prefix = '/nobackup/'

length = 64
# the projected features for id training data., the last dimension is the label for the feature.
#id_train_data = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro_maha_train.npy')
#id_train_data = torch.from_numpy(id_train_data).reshape(-1, length + 1)
#labels = id_train_data[:, -1].int()
#id_train_data = id_train_data[:, :-1]
# id_train_data = pickle.load(open('./data/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_train/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# labels = id_train_data['predicted_cls_id']
# id_train_data = torch.stack(id_train_data['binary_cls'])

# the projected features for id validation/test data.
#all_data_in = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/id-pro.npy')
#all_data_in = torch.from_numpy(all_data_in).reshape(-1, length)
id_val_data = pickle.load(open(prefix + 'my_xfdu/BDD-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/bdd_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# labels_val = id_val_data['predicted_cls_id']
id_val_data = torch.stack(id_val_data['binary_cls']).cpu()

## the projected features for ood data.
# all_data_out = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + name + '/ood-pro.npy')
# all_data_out = torch.from_numpy(all_data_out).reshape(-1, length)
if args.open:
    ood_val_data = pickle.load(open(
        prefix + 'my_xfdu/BDD-Detection/' + args.model + '/' + args.name + '/random_seed' + '_' + str(
            args.seed) + '/inference/openimages_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_' + str(
            args.thres) + '.pkl', 'rb'))
else:
    ood_val_data = pickle.load(open(prefix + 'my_xfdu/BDD-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/coco_ood_val_bdd/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# labels_ood = ood_val_data['predicted_cls_id']
ood_val_data = torch.stack(ood_val_data['binary_cls']).cpu()


id = 0
T = 1
id_score = []
ood_score = []

mean_list = []

# id_train_data= id_train_data.cpu().numpy()
# class_id = labels.cpu().numpy()
#
# data_train = id_train_data / np.linalg.norm(id_train_data, ord=2, axis=-1, keepdims=True)
# mean_class = np.zeros((20, length))
# # class_id = labels #labels.reshape(-1,1)
# # data_train = np.concatenate([data_train, class_id], 1)
# data_train = np.hstack((data_train, np.expand_dims(class_id, 1)))
# sample_dict = {}
#
# for i in range(20):
#     sample_dict[i] = []
# for data in data_train:
#     mean_class[int(data[-1])] += data[:-1]
#     sample_dict[int(data[-1])].append(data[:-1])
#
# for i in range(20):
#     mean_class[i] = mean_class[i] / len(sample_dict[i])
# mean_class = torch.from_numpy(mean_class)
# mean_list.append(mean_class)

#print(len(all_data_out))
#print(len(all_data_in))
print(len(id_val_data))
print(len(ood_val_data))



# from vMF import density

if args.use_es:
    mean_load = np.load(prefix + 'my_xfdu/BDD-Detection/' + args.model + '/'+args.name +'/proto.npy')
    kappa_load = np.load(prefix + 'my_xfdu/BDD-Detection/' + args.model + '/'+args.name  +'/kappa.npy')
    # kappa_load = np.load('/nobackup-slow/dataset/my_xfdu/detr_out/exps/' + args.name + '/kappa.npy')
    print(kappa_load)

gaussian_score = 0
gaussian_score1 = 0
for i in range(10):
    # breakpoint()
    if args.use_es == 0:
        batch_sample_mean = mean_list[-1][i]
        xm_norm = (batch_sample_mean ** 2).sum().sqrt()
        mu0 = batch_sample_mean / xm_norm
        kappa0 = (len(batch_sample_mean) * xm_norm - xm_norm ** 3) / (1 - xm_norm ** 2)

        # prob_density = density(mu0.numpy(), kappa0.numpy(), F.normalize(all_data_in, p=2, dim=-1).numpy())
        prob_density = density(mu0.numpy(), kappa0.numpy(), F.normalize(id_val_data, p=2, dim=-1).numpy())
        # prob_density1 = density(mu0.numpy(), kappa0.numpy(), F.normalize(all_data_out, p=2, dim=-1).numpy())
        prob_density1 = density(mu0.numpy(), kappa0.numpy(), F.normalize(ood_val_data, p=2, dim=-1).numpy())
    else:
        mu0 = mean_load[i]
        kappa0 = kappa_load[0][i]
        # prob_density = density(mu0, kappa0, F.normalize(all_data_in, p=2, dim=-1).numpy())
        prob_density = density(mu0, kappa0, F.normalize(id_val_data, p=2, dim=-1).numpy())
        # prob_density1 = density(mu0, kappa0, F.normalize(all_data_out, p=2, dim=-1).numpy())
        prob_density1 = density(mu0, kappa0, F.normalize(ood_val_data, p=2, dim=-1).numpy())
    # breakpoint()
    if i == 0:
        gaussian_score = prob_density.view(-1,1)
        gaussian_score1 = prob_density1.view(-1,1)
    else:
        gaussian_score = torch.cat((gaussian_score, prob_density.view(-1,1)), 1)
        gaussian_score1 = torch.cat((gaussian_score1, prob_density1.view(-1, 1)), 1)

id_score, _ = torch.max(gaussian_score, dim=1)
ood_score, _ = torch.max(gaussian_score1, dim=1)

#print(len(id_score))
#print(len(ood_score))
print(len(id_val_data))
print(len(ood_val_data))

measures = get_measures(id_score.cpu().data.numpy(), ood_score.cpu().data.numpy(), plot=False)

# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'energy')




