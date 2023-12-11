import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2,num_channels = 3):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.num_channels = num_channels
        self.enc = nn.Encoder(self.z_dim, num_channels = self.num_channels)
        self.dec = nn.Decoder(self.z_dim, num_channels = self.num_channels)
        self.pixels = 256
        self.batch_size = 10
        self.beta = 0.01
        self.beta_start = 0.01
        self.beta_end = 2
        self.show_stats = False

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################

        ################################################################################
        # End of code modification
        ################################################################################
        
        m,v = self.enc(x)
        kl = ut.kl_normal(m,v,self.z_prior[0].expand(m.shape),self.z_prior[1].expand(v.shape))
        #sample latent variables
        z = ut.sample_gaussian(m,v)
        #pass z through nerual network decoder to obtain logits of the d bernoulli random vairables
        logits = self.dec(z)
        #get the log bernoulli of logits

        #10 should be softcoded to batch size
        pixels = self.pixels
    
        #rec = ut.mse_loss(x.view(self.batch_size, 3, pixels*pixels), logits.view(self.batch_size, 3, pixels*pixels))
        rec = ut.log_cosh_loss(x.view(self.batch_size, 3, pixels*pixels), logits.view(self.batch_size, 3, pixels*pixels))

        kl = torch.mean(kl)
        rec = -torch.mean(rec)

        nelbo = rec + (self.beta * kl)
        


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

        ################################################################################
        # End of code modification
        ################################################################################
        batch = x.shape[0]
        mult_x = ut.duplicate(x,iw)
        m,v = self.enc(x)
        mult_m = ut.duplicate(m,iw)
        mult_v = ut.duplicate(v,iw)
        z = ut.sample_gaussian(mult_m,mult_v)
        
        #pass z through nerual network decoder to botain logits of the d bernoulli random vairables
        mult_pm = self.z_prior[0].expand(mult_m.shape)
        mult_pv = self.z_prior[1].expand(mult_v.shape)
        logits = self.dec(z)
        rec = ut.log_bernoulli_with_logits(mult_x,logits)
        log_ratios = ut.log_normal(z,mult_pm,mult_pv) + rec - ut.log_normal(z,mult_m,mult_v)
        log_ratios = log_ratios.reshape(iw,batch)
        niwae = ut.log_mean_exp(log_ratios,0)
        niwae = -torch.mean(niwae)

        pm = self.z_prior[0].expand(m.shape)
        pv = self.z_prior[1].expand(v.shape)
        kls = ut.kl_normal(m, v, pm, pv)
        kl = torch.mean(kls)
        rec = -torch.mean(rec)
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec)
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        logits = self.dec(z)
        return logits
    
    def get_num_pixels(self):
        return self.pixels
    

