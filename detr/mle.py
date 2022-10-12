import torch
from torch.utils.data import DataLoader

from vmf1 import *
import utils


'''
Define a true vMF model
'''

mu_true = torch.zeros(20)  # 5 or 20
mu_true[0] = 1.0
mu_true = mu_true / utils.norm(mu_true, dim=0)
kappa_true = torch.tensor(500.0)  # 50.0
vmf_true = vMF(x_dim=mu_true.shape[0])
vmf_true.set_params(mu=mu_true, kappa=kappa_true)
vmf_true = vmf_true.cuda()

# sample from true vMF model
samples = vmf_true.sample(N=1000, rsf=1)


'''
Full-batch ML estimator
'''

xm = samples.mean(0)
xm_norm = (xm**2).sum().sqrt()
mu0 = xm / xm_norm
kappa0 = (len(xm)*xm_norm - xm_norm**3) / (1-xm_norm**2)

mu_err = ((mu0.cpu() - mu_true)**2).sum().item()  # relative error
kappa_err = (kappa0.cpu() - kappa_true).abs().item() / kappa_true.item()
prn_str = '== Batch ML estimator ==\n'
prn_str += 'mu = %s (error = %.8f)\n' % (mu0.cpu().numpy(), mu_err)
prn_str += 'kappa = %s (error = %.8f)\n' % (kappa0.cpu().numpy(), kappa_err)
print(prn_str)
print(kappa0)
breakpoint()