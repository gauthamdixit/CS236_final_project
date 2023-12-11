# Copyright (c) 2021 Rui Shu
import numpy as np
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=16, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)
        self.pixels = 128
        self.batch_size = 16
        self.beta = 0
        self.beta_start = 0
        self.beta_end = 1

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        prior_m,prior_v = prior
        batch = x.shape[0]
        m,v = self.enc(x)
        z = ut.sample_gaussian(m,v)
        multi_m = prior_m.expand(batch, *prior_m.shape[1:])
        multi_v = prior_v.expand(batch, *prior_v.shape[1:])
        kl = ut.log_normal(z,m,v) - ut.log_normal_mixture(z,multi_m,multi_v)
        #sample latent variables
        pixels = self.pixels
        
        #pass z through nerual network decoder to botain logits of the d bernoulli random vairables
        logits = self.dec(z)
        #get the log bernoulli of logits
        #batch variable already exists so consider changing later
        x_view = x.view(self.batch_size, 3, pixels*pixels)
        log_view = logits.view(self.batch_size, 3, pixels*pixels)
        rec = ut.log_cosh_loss(x_view,log_view)
        #rec = ut.mse_loss(x_view,log_view)

        kl = torch.mean(kl)
        rec = -torch.mean(rec)
        nelbo = rec + (self.beta*kl)
        #nelbo on test subset should be around 100

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        

        ################################################################################
        # End of code modification
        ################################################################################
        prior = ut.gaussian_parameters(self.z_pre, dim=1)
        batch = x.shape[0]
        mult_x = ut.duplicate(x,iw)
        m,v = self.enc(x)
        mult_qm = ut.duplicate(m,iw)
        mult_qv = ut.duplicate(v,iw)
        z = ut.sample_gaussian(mult_qm,mult_qv)
        
        #pass z through nerual network decoder to botain logits of the d bernoulli random vairables
        multi_m = prior[0].expand(batch*iw, *prior[0].shape[1:])
        multi_v = prior[1].expand(batch*iw, *prior[1].shape[1:])
        logits = self.dec(z)
        rec = ut.log_bernoulli_with_logits(mult_x,logits)
        log_mix = ut.log_normal_mixture(z, multi_m, multi_v)
        log_norm = ut.log_normal(z,mult_qm,mult_qv)
        log_ratios = log_mix + rec - log_norm
        log_ratios = log_ratios.reshape(iw,batch)
        niwae = ut.log_mean_exp(log_ratios,0)
        niwae = -torch.mean(niwae)

        kl = log_norm - log_mix
        kl = torch.mean(kl)
        rec = -torch.mean(rec)
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries


    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
         logits = self.dec(z)
         return logits

    def get_num_pixels(self):
        return self.pixels