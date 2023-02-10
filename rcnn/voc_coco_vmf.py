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
parser.add_argument('--length', default=64, type=int)
parser.add_argument('--name', default='center64_0.1', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default='faster-rcnn', type=str)
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
prefix = '/nobackup/dataset/'
length = args.length

id_val_data = pickle.load(open(prefix + 'my_xfdu/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/voc_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
id_val_data = torch.stack(id_val_data['projections']).cpu()


if args.open:
    ood_val_data = pickle.load(open(
        prefix + 'my_xfdu/VOC-Detection/' + args.model + '/' + args.name + '/random_seed' + '_' + str(
            args.seed) + '/inference/openimages_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_' + str(
            args.thres) + '.pkl', 'rb'))
else:
    ood_val_data = pickle.load(open(prefix + 'my_xfdu/VOC-Detection/' + args.model + '/'+args.name+'/random_seed'+'_' +str(args.seed)  +'/inference/coco_ood_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
# labels_ood = ood_val_data['predicted_cls_id']
ood_val_data = torch.stack(ood_val_data['projections']).cpu()


id = 0
T = 1
id_score = []
ood_score = []

mean_list = []


print(len(id_val_data))
print(len(ood_val_data))



# from vMF import density


mean_load = np.load(prefix + 'my_xfdu/VOC-Detection/' + args.model + '/'+args.name +'/proto.npy')
kappa_load = np.load(prefix + 'my_xfdu/VOC-Detection/' + args.model + '/'+args.name  +'/kappa.npy')

print(kappa_load)

gaussian_score = 0
gaussian_score1 = 0
for i in range(20):
    mu0 = mean_load[i]
    kappa0 = kappa_load[0][i]
    prob_density = density(mu0, kappa0, F.normalize(id_val_data, p=2, dim=-1).numpy())
    prob_density1 = density(mu0, kappa0, F.normalize(ood_val_data, p=2, dim=-1).numpy())

    if i == 0:
        gaussian_score = prob_density.view(-1,1)
        gaussian_score1 = prob_density1.view(-1,1)
    else:
        gaussian_score = torch.cat((gaussian_score, prob_density.view(-1,1)), 1)
        gaussian_score1 = torch.cat((gaussian_score1, prob_density1.view(-1, 1)), 1)

id_score, _ = torch.max(gaussian_score, dim=1)
ood_score, _ = torch.max(gaussian_score1, dim=1)


print(len(id_val_data))
print(len(ood_val_data))

measures = get_measures(id_score.cpu().data.numpy(), ood_score.cpu().data.numpy(), plot=False)

# if args.energy:
print_measures(measures[0], measures[1], measures[2], 'energy')




