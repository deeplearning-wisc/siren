import math
from torch.distributions.kl import register_kl
from .ive import ive, ive_fraction_approx, ive_fraction_approx2
from .hyperspherical_uniform import (
    HypersphericalUniform,
)

import numpy as np
import mpmath
import torch
import torch.nn as nn

from .utils import realmin, norm


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


class vMF(nn.Module):
    '''
    vMF(x; mu, kappa)
    '''

    def __init__(self, x_dim, reg=1e-6):

        super(vMF, self).__init__()

        self.x_dim = x_dim

        self.mu_unnorm = nn.Parameter(torch.randn(x_dim))
        self.logkappa = nn.Parameter(0.01 * torch.randn([]))

        self.reg = reg

    def set_params(self, mu, kappa):

        with torch.no_grad():
            self.mu_unnorm.copy_(mu)
            self.logkappa.copy_(torch.log(kappa + realmin))

    def get_params(self):

        mu = self.mu_unnorm / norm(self.mu_unnorm)
        kappa = self.logkappa.exp() + self.reg

        return mu, kappa

    def forward(self, x, utc=False):

        '''
        Evaluate logliks, log p(x)

        Args:
            x = batch for x
            utc = whether to evaluate only up to constant or exactly
                if True, no log-partition computed
                if False, exact loglik computed
        Returns:
            logliks = log p(x)
        '''

        mu, kappa = self.get_params()

        dotp = (mu.unsqueeze(0) * x).sum(1)

        if utc:
            logliks = kappa * dotp
        else:
            logC = vMFLogPartition.apply(self.x_dim, kappa)
            logliks = kappa * dotp + logC

        return logliks

    def sample(self, N=1, rsf=10):

        '''
        Args:
            N = number of samples to generate
            rsf = multiplicative factor for extra backup samples in rejection sampling

        Returns:
            samples; N samples generated

        Notes:
            no autodiff
        '''

        d = self.x_dim

        with torch.no_grad():

            mu, kappa = self.get_params()

            # Step-1: Sample uniform unit vectors in R^{d-1}
            v = torch.randn(N, d - 1).to(mu)
            v = v / norm(v, dim=1)

            # Step-2: Sample v0
            kmr = np.sqrt(4 * kappa.item() ** 2 + (d - 1) ** 2)
            bb = (kmr - 2 * kappa) / (d - 1)
            aa = (kmr + 2 * kappa + d - 1) / 4
            dd = (4 * aa * bb) / (1 + bb) - (d - 1) * np.log(d - 1)
            beta = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)))
            uniform = torch.distributions.Uniform(0.0, 1.0)
            v0 = torch.tensor([]).to(mu)
            while len(v0) < N:
                eps = beta.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
                uns = uniform.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
                w0 = (1 - (1 + bb) * eps) / (1 - (1 - bb) * eps)
                t0 = (2 * aa * bb) / (1 - (1 - bb) * eps)
                det = (d - 1) * t0.log() - t0 + dd - uns.log()
                copy_tensor = (w0[det >= 0]).clone().detach()
                v0 = torch.cat([v0, copy_tensor.to(mu)])
                if len(v0) > N:
                    v0 = v0[:N]
                    break
            v0 = v0.reshape([N, 1])

            # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
            samples = torch.cat([v0, (1 - v0 ** 2).sqrt() * v], 1)

            # Setup-4: Householder transformation
            e1mu = torch.zeros(d, 1).to(mu);
            e1mu[0, 0] = 1.0
            e1mu = e1mu - mu if len(mu.shape) == 2 else e1mu - mu.unsqueeze(1)
            e1mu = e1mu / norm(e1mu, dim=0)
            samples = samples - 2 * (samples @ e1mu) @ e1mu.t()

        return samples



# class VonMisesFisher(torch.distributions.Distribution):
#
#     arg_constraints = {
#         "loc": torch.distributions.constraints.real,
#         "scale": torch.distributions.constraints.positive,
#     }
#     support = torch.distributions.constraints.real
#     has_rsample = True
#     _mean_carrier_measure = 0
#
#     @property
#     def mean(self):
#         # option 1:
#         return self.loc * (
#             ive(self.__m / 2, self.scale) / ive(self.__m / 2 - 1, self.scale)
#         )
#         # option 2:
#         # return self.loc * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
#         # options 3:
#         # return self.loc * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)
#
#     @property
#     def stddev(self):
#         return self.scale
#
#     def __init__(self, loc, scale, validate_args=None, k=1):
#         self.dtype = loc.dtype
#         self.loc = loc
#         self.scale = scale
#         self.device = loc.device
#         self.__m = loc.shape[-1]
#         self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(self.device)
#         self.k = k
#
#         super().__init__(self.loc.size(), validate_args=validate_args)
#
#     def sample(self, shape=torch.Size()):
#         with torch.no_grad():
#             return self.rsample(shape)
#
#     def rsample(self, shape=torch.Size()):
#         shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
#
#         w = (
#             self.__sample_w3(shape=shape)
#             if self.__m == 3
#             else self.__sample_w_rej(shape=shape)
#         )
#
#         v = (
#             torch.distributions.Normal(0, 1)
#             .sample(shape + torch.Size(self.loc.shape))
#             .to(self.device)
#             .transpose(0, -1)[1:]
#         ).transpose(0, -1)
#         v = v / v.norm(dim=-1, keepdim=True)
#
#         w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
#         x = torch.cat((w, w_ * v), -1)
#         z = self.__householder_rotation(x)
#
#         return z.type(self.dtype)
#
#     def __sample_w3(self, shape):
#         shape = shape + torch.Size(self.scale.shape)
#         u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
#         self.__w = (
#             1
#             + torch.stack(
#                 [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
#             ).logsumexp(0)
#             / self.scale
#         )
#         return self.__w
#
#     def __sample_w_rej(self, shape):
#         c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
#         b_true = (-2 * self.scale + c) / (self.__m - 1)
#
#         # using Taylor approximation with a smooth swift from 10 < scale < 11
#         # to avoid numerical errors for large scale
#         b_app = (self.__m - 1) / (4 * self.scale)
#         s = torch.min(
#             torch.max(
#                 torch.tensor([0.0], dtype=self.dtype, device=self.device),
#                 self.scale - 10,
#             ),
#             torch.tensor([1.0], dtype=self.dtype, device=self.device),
#         )
#         b = b_app * s + b_true * (1 - s)
#
#         a = (self.__m - 1 + 2 * self.scale + c) / 4
#         d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)
#
#         self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
#         return self.__w
#
#     @staticmethod
#     def first_nonzero(x, dim, invalid_val=-1):
#         mask = x > 0
#         idx = torch.where(
#             mask.any(dim=dim),
#             mask.float().argmax(dim=1).squeeze(),
#             torch.tensor(invalid_val, device=x.device),
#         )
#         return idx
#
#     def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
#         #  matrix while loop: samples a matrix of [A, k] samples, to avoid looping all together
#         b, a, d = [
#             e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
#             for e in (b, a, d)
#         ]
#         w, e, bool_mask = (
#             torch.zeros_like(b).to(self.device),
#             torch.zeros_like(b).to(self.device),
#             (torch.ones_like(b) == 1).to(self.device),
#         )
#
#         sample_shape = torch.Size([b.shape[0], k])
#         shape = shape + torch.Size(self.scale.shape)
#
#         while bool_mask.sum() != 0:
#             con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
#             con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
#             e_ = (
#                 torch.distributions.Beta(con1, con2)
#                 .sample(sample_shape)
#                 .to(self.device)
#                 .type(self.dtype)
#             )
#
#             u = (
#                 torch.distributions.Uniform(0 + eps, 1 - eps)
#                 .sample(sample_shape)
#                 .to(self.device)
#                 .type(self.dtype)
#             )
#
#             w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
#             t = (2 * a * b) / (1 - (1 - b) * e_)
#
#             accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
#             accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
#             accept_idx_clamped = accept_idx.clamp(0)
#             # we use .abs(), in order to not get -1 index issues, the -1 is still used afterwards
#             w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
#             e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))
#
#             reject = accept_idx < 0
#             accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject
#
#             w[bool_mask * accept] = w_[bool_mask * accept]
#             e[bool_mask * accept] = e_[bool_mask * accept]
#
#             bool_mask[bool_mask * accept] = reject[bool_mask * accept]
#
#         return e.reshape(shape), w.reshape(shape)
#
#     def __householder_rotation(self, x):
#         u = self.__e1 - self.loc
#         u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
#         z = x - 2 * (x * u).sum(-1, keepdim=True) * u
#         return z
#
#     def entropy(self):
#         # option 1:
#         output = (
#             -self.scale
#             * ive(self.__m / 2, self.scale)
#             / ive((self.__m / 2) - 1, self.scale)
#         )
#         # option 2:
#         # output = - self.scale * ive_fraction_approx(torch.tensor(self.__m / 2), self.scale)
#         # option 3:
#         # output = - self.scale * ive_fraction_approx2(torch.tensor(self.__m / 2), self.scale)
#
#         return output.view(*(output.shape[:-1])) + self._log_normalization()
#
#     def log_prob(self, x):
#         return self._log_unnormalized_prob(x) - self._log_normalization()
#
#     def _log_unnormalized_prob(self, x):
#         output = self.scale * (self.loc * x).sum(-1, keepdim=True)
#
#         return output.view(*(output.shape[:-1]))
#
#     def _log_normalization(self):
#         output = -(
#             (self.__m / 2 - 1) * torch.log(self.scale)
#             - (self.__m / 2) * math.log(2 * math.pi)
#             - (self.scale + torch.log(ive(self.__m / 2 - 1, self.scale)))
#         )
#
#         return output.view(*(output.shape[:-1]))
#
#
# @register_kl(VonMisesFisher, HypersphericalUniform)
# def _kl_vmf_uniform(vmf, hyu):
#     return -vmf.entropy() + hyu.entropy()