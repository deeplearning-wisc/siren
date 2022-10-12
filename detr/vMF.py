'''
Generate multivariate von Mises Fisher samples.
This solution originally appears here:
http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
Also see:
Sampling from vMF on S^2:
    https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf
This code was taken from the following project:
https://github.com/clara-labs/spherecluster
'''
import numpy as np
import os
import torch
import mpmath
import torch.nn as nn
import utils


# __all__ = ['sample_vMF']


# realmin = 1e-10
#
#
# def norm(input, p=2, dim=0, eps=1e-12):
#     return input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)
#
#
# def mkdirs(path):
#     if not os.path.exists(path):
#         os.makedirs(path)


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


def sample_vMF(mu, kappa, num_samples):
    # breakpoint()
    rsf=1
    d = len(mu)
    mu = torch.from_numpy(mu).float()
    #
    kappa = torch.tensor(kappa)
    N = num_samples
    with torch.no_grad():

        # mu, kappa = self.get_params()
        # Step-1: Sample uniform unit vectors in R^{d-1}
        v = torch.randn(N, d - 1).to(mu)
        v = v / utils.norm(v, dim=1)
        # breakpoint()
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
            v0 = torch.cat([v0, torch.tensor(w0[det >= 0]).to(mu)])
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
        e1mu = e1mu / utils.norm(e1mu, dim=0)
        samples = samples - 2 * (samples @ e1mu) @ e1mu.t()
    samples = samples.numpy()
    # breakpoint()
    return samples

# def sample_vMF(mu, kappa, num_samples):
#     """Generate num_samples N-dimensional samples from von Mises Fisher
#     distribution around center mu \in R^N with concentration kappa.
#     """
#     dim = len(mu)
#     result = np.zeros((num_samples, dim))
#     for nn in range(num_samples):
#         # sample offset from center (on sphere) with spread kappa
#         w = _sample_weight(kappa, dim)
#
#         # sample a point v on the unit sphere that's orthogonal to mu
#         v = _sample_orthonormal_to(mu)
#
#         # compute new point
#         result[nn, :] = v * np.sqrt(1. - w**2) + w * mu
#
#     return result
    # # a new version.
    #
    # d = len(mu)
    # N = num_samples
    # # Step-1: Sample uniform unit vectors in R^{d-1}
    # v = torch.randn(N, d - 1).to(mu)
    # v = v / norm(v, dim=1)
    #
    # # Step-2: Sample v0
    # kmr = np.sqrt(4 * kappa.item() ** 2 + (d - 1) ** 2)
    # bb = (kmr - 2 * kappa) / (d - 1)
    # aa = (kmr + 2 * kappa + d - 1) / 4
    # dd = (4 * aa * bb) / (1 + bb) - (d - 1) * np.log(d - 1)
    # beta = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)))
    # uniform = torch.distributions.Uniform(0.0, 1.0)
    # v0 = torch.tensor([]).to(mu)
    # while len(v0) < N:
    #     eps = beta.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
    #     uns = uniform.sample([1, rsf * (N - len(v0))]).squeeze().to(mu)
    #     w0 = (1 - (1 + bb) * eps) / (1 - (1 - bb) * eps)
    #     t0 = (2 * aa * bb) / (1 - (1 - bb) * eps)
    #     det = (d - 1) * t0.log() - t0 + dd - uns.log()
    #     v0 = torch.cat([v0, torch.tensor(w0[det >= 0]).to(mu)])
    #     if len(v0) > N:
    #         v0 = v0[:N]
    #         break
    # v0 = v0.reshape([N, 1])
    #
    # # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
    # samples = torch.cat([v0, (1 - v0 ** 2).sqrt() * v], 1)
    #
    # # Setup-4: Householder transformation
    # e1mu = torch.zeros(d, 1).to(mu);
    # e1mu[0, 0] = 1.0
    # e1mu = e1mu - mu if len(mu.shape) == 2 else e1mu - mu.unsqueeze(1)
    # e1mu = e1mu / utils.norm(e1mu, dim=0)
    # samples = samples - 2 * (samples @ e1mu) @ e1mu.t()
    #
    # return samples


# def _sample_weight(kappa, dim):
#     """Rejection sampling scheme for sampling distance from center on
#     surface of the sphere.
#     """
#     dim = dim - 1  # since S^{n-1}
#     b = dim / (np.sqrt(4. * kappa**2 + dim**2) + 2 * kappa)
#     x = (1. - b) / (1. + b)
#     c = kappa * x + dim * np.log(1 - x**2)
#
#     while True:
#         z = np.random.beta(dim / 2., dim / 2.)
#         w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
#         u = np.random.uniform(low=0, high=1)
#         if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
#             return w
#
#
# def _sample_orthonormal_to(mu):
#     """Sample point on sphere orthogonal to mu."""
#     v = np.random.randn(mu.shape[0])
#     proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
#     orthto = v - proj_mu_v
#     return orthto / np.linalg.norm(orthto)