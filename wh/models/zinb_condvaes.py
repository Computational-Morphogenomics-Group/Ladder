########################################################
### CondVAE Based  Models - ZINB reconstruction ########
######################################################## 

import torch, os, sys, time

import torch.nn as nn

import torch.distributions as dists
import torch.nn.functional as F
from .basics import MLP
import numpy as np


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================


class ZINB_CSVAE(nn.Module):

    "CSVAE, learned prior, ZINB reconstruction."
    
    def __init__(self, input_size=3, label_size=2, common_latent_size=2, weighted_latent_size=2, enc_sizes=None, dec_sizes=None, mlp_hidden=64, mlp_hidden_count=3, mlp_bias=True, betas=[20,1,0.2,10,1], distribution='zinb'):
        
        super(ZINB_CSVAE, self).__init__()

        if enc_sizes is None:
            enc_sizes = [mlp_hidden]*mlp_hidden_count

        if dec_sizes is None:
            dec_sizes = [mlp_hidden]*mlp_hidden_count
        
        # Define latent sizes
        self.x_size, self.y_size, self.xy_size, self.z_size, self.w_size, self.single_w_size = input_size, label_size, input_size + label_size, common_latent_size, weighted_latent_size*label_size, weighted_latent_size

        # Define additional hyperparams
        self.betas, self.distribution = betas, distribution

        
        # Encoding
        
        ## W variational params
        self.w_latent = MLP(input_size=self.xy_size, hidden_sizes=enc_sizes, output_size=self.w_size, bias=mlp_bias)
        self.mu_w = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size, bias=mlp_bias)
        self.logvar_w = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size, bias=mlp_bias)
        
        ## Z variational params
        self.z_latent = MLP(input_size=self.x_size, hidden_sizes=enc_sizes, output_size=self.z_size, bias=mlp_bias)
        self.mu_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size, bias=mlp_bias)
        self.logvar_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size, bias=mlp_bias)

        ## W prior variational params
        self.w_prior = MLP(input_size=self.y_size, hidden_sizes=enc_sizes, output_size=self.w_size, bias=mlp_bias)
        self.mu_w_prior = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size, bias=mlp_bias)
        self.logvar_w_prior = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size, bias=mlp_bias)
        
        
        # Decoding
        
        ## Reconstruction variational params
        self.x_latent = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[self.z_size + self.w_size]*mlp_hidden_count, output_size=self.z_size+self.w_size, bias=mlp_bias)
        self.mu_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=dec_sizes, output_size=self.x_size, bias=mlp_bias)
        self.logits_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=dec_sizes, output_size=self.x_size, final_activation=nn.Softplus(), bias=mlp_bias)

        ## ZINB theta (kept in log)
        self.log_theta = torch.nn.Parameter(torch.randn(self.x_size))
        
        ## Mutual information minimizer 
        self.z_y = MLP(input_size=self.z_size, hidden_sizes=([self.z_size]*2) + ([mlp_hidden]*mlp_hidden_count), output_size=self.y_size, final_activation=nn.Sigmoid(), bias=mlp_bias)

    
    
    # Encode
    def xy_zw(self, x, y):
        
        # Concat inputs      
        xy = torch.cat([x, y], dim=1)

        # Generate w and z params
        w_latent = self.w_latent(xy)
        w_mu = self.mu_w(w_latent)
        w_logvar = self.logvar_w(w_latent)

    
        z_latent = self.z_latent(x)
        z_mu = self.mu_z(z_latent)
        z_logvar = self.logvar_z(z_latent)

        w_prior_latent = self.w_prior(y)
        w_prior_mu = self.mu_w_prior(w_prior_latent)
        w_prior_logvar = self.logvar_w_prior(w_prior_latent)
    

        
        return w_mu, w_logvar, w_prior_mu, w_prior_logvar, z_mu, z_logvar
    
    
    # Reparam Trick
    @staticmethod
    def reparam(mu, logvar):
        
        std = logvar.mul(0.5).exp_()            
        eps = torch.DoubleTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)


    # Reconstruction
    def zw_x(self, zw):

        # Generate x params
        x_latent = self.x_latent(zw)
        mu_x = torch.exp(self.mu_x(x_latent))
        dropout_logits_x = self.logits_x(x_latent)
        
        return mu_x, dropout_logits_x
    

    # Run model
    def calc_interim(self, x, y):

        # Calculate variational params
        w_mu, w_logvar, w_prior_mu, w_prior_logvar, z_mu, z_logvar = self.xy_zw(x, y)


        # Calculate reparam vectors
        w_obs = self.reparam(w_mu, w_logvar)
        z = self.reparam(z_mu, z_logvar)
        zw = torch.cat([z, w_obs], dim=1)
    
        x_mu, x_dropout_logits = self.zw_x(zw)
        
        # Reconstruct y        
        y_pred = torch.clamp(self.z_y(z), min=1e-12, max=1 - 1e-12)
        
        return x_mu, x_dropout_logits, zw, y_pred, \
               w_mu, w_logvar, w_prior_mu,  \
               w_prior_logvar, z_mu, z_logvar


    # Loss to be minimized = M1 + M2
    def forward(self, x, y):

        
        # Run model 
        x_mu, x_dropout_logits, zw, y_pred, \
        w_mu, w_logvar, w_prior_mu,  \
        w_prior_logvar, z_mu, z_logvar = self.calc_interim(torch.log1p(x), y)


        # Get loss components
        
        ## ELBO
        ### X reconstruction - ZINB
        theta = self.log_theta.exp()
        nb_logits = (x_mu+1e-12).log() - (theta+1e-12).log()
        
        if self.distribution == 'zinb':
            distribution = ZeroInflatedNegativeBinomial(total_count=theta, logits=nb_logits, gate_logits = x_dropout_logits, validate_args=False)
        
        elif self.distribution == 'nb':
            distribution = NegativeBinomial(total_count=theta, logits=nb_logits, validate_args=False)
        
        x_recon_loss = -1 * distribution.log_prob(x).sum(-1).mean() * self.betas[0]
        
    
        ### KL Div - W 
        w_obs_dist = dists.MultivariateNormal(w_mu.flatten(), torch.diag(w_logvar.flatten().exp()))
        w_prior_dist = dists.MultivariateNormal(w_prior_mu.flatten(), torch.diag(w_prior_logvar.flatten().exp()))
        w_kl = dists.kl.kl_divergence(w_obs_dist, w_prior_dist) * self.betas[1]

        ### KL Div - Z - Prior N(0,I)
        z_obs_dist = dists.MultivariateNormal(z_mu.flatten(), torch.diag(z_logvar.flatten().exp()))
        z_prior_dist = dists.MultivariateNormal(torch.zeros(self.z_size * z_mu.size()[0]).to(x.device), torch.eye(self.z_size * z_mu.size()[0]).to(x.device)) # Might need to change this if not batching
        z_kl = dists.kl.kl_divergence(z_obs_dist, z_prior_dist) * self.betas[2]

        ## Minimize MI between Y&Z
        y_pred_negentropy = ( (y_pred.log() * y_pred) + ((1-y_pred).log() * (1-y_pred)) ).mean() * self.betas[3]
        y_recon = F.binary_cross_entropy(y_pred, y) * self.betas[4] # Discriminator will learn this

        
        # Combine 
        M1_M2 = x_recon_loss + w_kl + z_kl + y_pred_negentropy


        return M1_M2, x_recon_loss, w_kl, z_kl, y_pred_negentropy, y_recon


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
