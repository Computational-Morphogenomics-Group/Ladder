####################################
### CondVAE Based  Models ##########
####################################

import torch, os, sys, time

import torch.nn as nn

import torch.distributions as dists
import torch.nn.functional as F
from .basics import MLP


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================


class CSVAE(nn.Module):
    """
    The CSVAE with a learnable prior from the labels, as defined in 'Learning Latent Subspaces in Variational Autoencoders'
    """

    # Define Params
    def __init__(self, input_size=3, label_size=2, common_latent_size=2, weighted_latent_size=2, mlp_hidden=64, mlp_hidden_count=3, betas=[20,1,0.2,10,1]):
        super(CSVAE, self).__init__()
        
        # Define latent sizes
        self.x_size, self.y_size, self.xy_size, self.z_size, self.w_size, self.single_w_size = input_size, label_size, input_size + label_size, common_latent_size, weighted_latent_size*label_size, weighted_latent_size

        # Define additional hyperparams
        self.betas = betas
        
        
        # Encoding
        
        ## W variational params
        self.w_latent = MLP(input_size=self.xy_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.w_size)
        self.mu_w = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size)
        self.logvar_w = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size)
        
        ## Z variational params
        self.z_latent = MLP(input_size=self.x_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.z_size)
        self.mu_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size)
        self.logvar_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size)

        ## W prior variational params
        self.w_prior = MLP(input_size=self.y_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.w_size)
        self.mu_w_prior = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size)
        self.logvar_w_prior = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size)
        
        
        # Decoding
        
        ## Reconstruction variational params
        self.x_latent = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.z_size+self.w_size)
        self.mu_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[self.z_size+self.w_size]*mlp_hidden_count, output_size=self.x_size)
        self.logvar_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[self.z_size+self.w_size]*mlp_hidden_count, output_size=self.x_size)
        
        ## Mutual information minimizer 
        self.z_y = MLP(input_size=self.z_size, hidden_sizes=([self.z_size]*2) + ([mlp_hidden]*mlp_hidden_count), output_size=self.y_size, final_activation=nn.Sigmoid())


    
    # Latent generation
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


    
    # Reconstruction
    def zw_x(self, zw):

        # Generate x params
        x_latent = self.x_latent(zw)
        mu_x = self.mu_x(x_latent)
        logvar_x = self.logvar_x(x_latent)
        
        return mu_x, logvar_x
    
    
    # Reparam Trick
    @staticmethod
    def reparam(mu, logvar):
        
        std = logvar.mul(0.5).exp_()            
        eps = torch.DoubleTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    
    
    # Run model
    def forward(self, x, y):

        # Calculate variational params
        w_mu, w_logvar, w_prior_mu, w_prior_logvar, z_mu, z_logvar = self.xy_zw(x, y)


        # Calculate reparam vectors
        w_obs = self.reparam(w_mu, w_logvar)
        z = self.reparam(z_mu, z_logvar)
        zw = torch.cat([z, w_obs], dim=1)
    
        x_mu, x_logvar = self.zw_x(zw)
        
        # Reconstruct y        
        y_pred = torch.clamp(self.z_y(z), min=1e-12, max=1 - 1e-12)
        
        return x_mu, x_logvar, zw, y_pred, \
               w_mu, w_logvar, w_prior_mu,  \
               w_prior_logvar, z_mu, z_logvar

    

    # Loss to be minimized = M1 + M2
    def M1_M2(self, x, y):
        
        # Run model 
        x_mu, x_logvar, zw, y_pred, \
        w_mu, w_logvar, w_prior_mu,  \
        w_prior_logvar, z_mu, z_logvar = self.forward(x,y)


        # Get loss components
        
        ## ELBO
        ### X reconstruction
        x_recon_loss = F.mse_loss(x_mu, x) * self.betas[0]

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
        



    # Generation - IN DEV
    def generate(self, x, y, hard_latent=False):
        
        # Map data to Z
        z_latent = self.z_latent(x)
        z_mu = self.mu_z(z_latent)
        z_logvar = self.logvar_z(z_latent)
        z = self.reparam(z_mu, z_logvar)

        # Sample a vector from W as described in paper
        w_prior_latent = self.w_prior(y)
        w_prior_mu = self.mu_w_prior(w_prior_latent)
        w_prior_logvar = self.logvar_w_prior(w_prior_latent)
        
        w = self.reparam(w_prior_mu, w_prior_logvar)


        # Make all other dimensions 0 optionally
        if hard_latent:
            w = torch.tensor([w[i][j] if lab[i][j//2] != 0 else 0 for i in range(w.shape[0]) for j in range(w.shape[1])]).reshape(w.shape).to(z.device)
            

        # Concat vectors and decode
        zw = torch.cat([z, w], dim=1)
        x_mu, x_logvar = self.zw_x(zw)
        x_gen = self.reparam(x_mu, x_logvar)
        
        return x_gen, y



#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================


class CSVAE_fixed_prior(nn.Module):
    """
    Same as CSVAE but the w prior is fixed and generated from y, as defined in 'Learning Latent Subspaces in Variational Autoencoders'
    """

    # Define Params
    def __init__(self, input_size=3, label_size=2, common_latent_size=2, weighted_latent_size=2, mlp_hidden=64, mlp_hidden_count=3, w_mus=[0,3], w_stds=[0.1,1] betas=[20,1,0.2,10,1]):
        super(CSVAE, self).__init__()
        
        # Define latent sizes
        self.x_size, self.y_size, self.xy_size, self.z_size, self.w_size, self.single_w_size = input_size, label_size, input_size + label_size, common_latent_size, weighted_latent_size*label_size, weighted_latent_size

        # Define additional hyperparams
        self.betas self.w_mus, self.w_stds = betas, w_mus, w_stds
        
        
        # Encoding
        
        ## W variational params
        self.w_latent = MLP(input_size=self.xy_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.w_size)
        self.mu_w = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size)
        self.logvar_w = MLP(input_size=self.w_size, hidden_sizes=[self.w_size]*mlp_hidden_count, output_size=self.w_size)
        
        ## Z variational params
        self.z_latent = MLP(input_size=self.x_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.z_size)
        self.mu_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size)
        self.logvar_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size)

        
        # Decoding
        
        ## Reconstruction variational params
        self.x_latent = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.z_size+self.w_size)
        self.mu_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[self.z_size+self.w_size]*mlp_hidden_count, output_size=self.x_size)
        self.logvar_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[self.z_size+self.w_size]*mlp_hidden_count, output_size=self.x_size)
        
        ## Mutual information minimizer 
        self.z_y = MLP(input_size=self.z_size, hidden_sizes=([self.z_size]*2) + ([mlp_hidden]*mlp_hidden_count), output_size=self.y_size, final_activation=nn.Sigmoid())


    
    # Latent generation
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

    

        
        return w_mu, w_logvar, z_mu, z_logvar


    
    # Reconstruction
    def zw_x(self, zw):

        # Generate x params
        x_latent = self.x_latent(zw)
        mu_x = self.mu_x(x_latent)
        logvar_x = self.logvar_x(x_latent)
        
        return mu_x, logvar_x

   
    # Fixed parameters for prior generation
    def w_prior(self, y):
        w_prior_mu = torch.DoubleTensor(np.array([(lambda num : [self.w_mus[1]]*(self.single_w_size) if num else [self.w_mus[0]]*(self.single_w_size))(num) for elem in y.detach() for num in elem])).reshape(y.shape[0], y.shape[1]*self.single_w_size).to(y.device)
        w_prior_var = torch.DoubleTensor(np.array([(lambda num : [self.w_stds[1]]*(self.single_w_size) if num else [self.w_stds[0]]*(self.single_w_size))(num) for elem in y.detach() for num in elem])).reshape(y.shape[0], y.shape[1]*self.single_w_size).to(y.device)

        w_prior_logvar = w_prior_var.log().div(0.5)
        
        return w_prior_mu, w_prior_logvar
    
    
    
    # Reparam Trick
    @staticmethod
    def reparam(mu, logvar):
        
        std = logvar.mul(0.5).exp_()            
        eps = torch.DoubleTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    
    
    # Run model
    def forward(self, x, y):

        # Calculate variational params
        w_mu, w_logvar, z_mu, z_logvar = self.xy_zw(x, y)
        w_prior_mu, w_prior_logvar = self.w_prior(y)
        

        # Calculate reparam vectors
        w_obs = self.reparam(w_mu, w_logvar)
        z = self.reparam(z_mu, z_logvar)
        zw = torch.cat([z, w_obs], dim=1)
    
        x_mu, x_logvar = self.zw_x(zw)
        
        # Reconstruct y        
        y_pred = torch.clamp(self.z_y(z), min=1e-12, max=1 - 1e-12)
        
        return x_mu, x_logvar, zw, y_pred, \
               w_mu, w_logvar, w_prior_mu,  \
               w_prior_logvar, z_mu, z_logvar

    

    # Loss to be minimized = M1 + M2
    def M1_M2(self, x, y):
        
        # Run model 
        x_mu, x_logvar, zw, y_pred, \
        w_mu, w_logvar, w_prior_mu,  \
        w_prior_logvar, z_mu, z_logvar = self.forward(x,y)


        # Get loss components
        
        ## ELBO
        ### X reconstruction
        x_recon_loss = F.mse_loss(x_mu, x) * self.betas[0]

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
        



    # Generation - IN DEV
    def generate(self, x, y, hard_latent=False):
        
        # Map data to Z
        z_latent = self.z_latent(x)
        z_mu = self.mu_z(z_latent)
        z_logvar = self.logvar_z(z_latent)
        z = self.reparam(z_mu, z_logvar)

        # Sample a vector from W as described in paper
        w_prior_mu, w_prior_logvar = self.w_prior(y)
        
        w = self.reparam(w_prior_mu, w_prior_logvar)


        # Make all other dimensions 0 optionally
        if hard_latent:
            w = torch.tensor([w[i][j] if lab[i][j//2] != 0 else 0 for i in range(w.shape[0]) for j in range(w.shape[1])]).reshape(w.shape).to(z.device)
            

        # Concat vectors and decode
        zw = torch.cat([z, w], dim=1)
        x_mu, x_logvar = self.zw_x(zw)
        x_gen = self.reparam(x_mu, x_logvar)
        
        return x_gen, y


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================



class CondVAE_Info(nn.Module):
    """
    Conditional VAE with MI minimization between z and label.
    """

    # Define Params
    def __init__(self, input_size=3, label_size=2, common_latent_size=2, mlp_hidden=64, mlp_hidden_count=3, betas=[20,0.2,10,1]):
        super(CSVAE, self).__init__()
        
        # Define latent sizes
        self.x_size, self.y_size, self.xy_size, self.z_size = input_size, label_size, input_size + label_size, common_latent_size

        # Define additional hyperparams
        self.betas = betas
        
        
        # Encoding
        
        ## Z variational params
        self.z_latent = MLP(input_size=self.x_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.z_size)
        self.mu_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size)
        self.logvar_z = MLP(input_size=self.z_size, hidden_sizes=[self.z_size]*mlp_hidden_count, output_size=self.z_size)


        
        # Decoding
        
        ## Reconstruction variational params
        self.x_latent = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[mlp_hidden]*mlp_hidden_count, output_size=self.z_size+self.w_size)
        self.mu_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[self.z_size + self.w_size]*mlp_hidden_count, output_size=self.x_size)
        self.logvar_x = MLP(input_size=self.z_size + self.w_size, hidden_sizes=[self.z_size + self.w_size]*mlp_hidden_count, output_size=self.x_size)
        
        ## Mutual information minimizer 
        self.z_y = MLP(input_size=self.z_size, hidden_sizes=([self.z_size]*2) + ([mlp_hidden]*mlp_hidden_count), output_size=self.y_size, final_activation=nn.Sigmoid())


    
    # Latent generation
    def x_z(self, x, y):
    
        # Generate z params
    
        z_latent = self.z_latent(x)
        z_mu = self.mu_z(z_latent)
        z_logvar = self.logvar_z(z_latent)

        
        return z_mu, z_logvar


    
    # Reconstruction
    def zy_x(self, zy):

        # Generate x params
        x_latent = self.x_latent(zy)
        mu_x = self.mu_x(x_latent)
        logvar_x = self.logvar_x(x_latent)
        
        return mu_x, logvar_x
    
    
    # Reparam Trick
    @staticmethod
    def reparam(mu, logvar):
        
        std = logvar.mul(0.5).exp_()            
        eps = torch.DoubleTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)

    
    
    # Run model
    def forward(self, x, y):

        # Calculate variational params
        z_mu, z_logvar = self.xy_zw(x, y)


        # Calculate reparam vectors
        z = self.reparam(z_mu, z_logvar)
        zy = torch.cat([z, y], dim=1)
    
        x_mu, x_logvar = self.zy_x(zy)
        
        # Reconstruct y        
        y_pred = torch.clamp(self.z_y(z), min=1e-12, max=1 - 1e-12)
        
        return x_mu, x_logvar, zy, y_pred, \
               z_mu, z_logvar

    

    # Loss to be minimized = M1 + M2
    def M1_M2(self, x, y):
        
        # Run model 
        x_mu, x_logvar, zw, y_pred, \
        z_mu, z_logvar = self.forward(x,y)


        # Get loss components
        
        ## ELBO
        ### X reconstruction
        x_recon_loss = F.mse_loss(x_mu, x) * self.betas[0]

        ### KL Div - Z - Prior N(0,I)
        z_obs_dist = dists.MultivariateNormal(z_mu.flatten(), torch.diag(z_logvar.flatten().exp()))
        z_prior_dist = dists.MultivariateNormal(torch.zeros(self.z_size * z_mu.size()[0]).to(x.device), torch.eye(self.z_size * z_mu.size()[0]).to(x.device)) # Might need to change this if not batching
        z_kl = dists.kl.kl_divergence(z_obs_dist, z_prior_dist) * self.betas[1]

        ## Minimize MI between Y&Z
        y_pred_negentropy = ( (y_pred.log() * y_pred) + ((1-y_pred).log() * (1-y_pred)) ).mean() * self.betas[2]
        y_recon = F.binary_cross_entropy(y_pred, y) * self.betas[3] # Discriminator will learn this

        
        # Combine 
        M1_M2 = x_recon_loss + z_kl + y_pred_negentropy


        return M1_M2, x_recon_loss, z_kl, y_pred_negentropy, y_recon
        



    # Generation - IN DEV
    def generate(self, x, y):
        
        # Map data to Z
        z_latent = self.z_latent(x)
        z_mu = self.mu_z(z_latent)
        z_logvar = self.logvar_z(z_latent)
        z = self.reparam(z_mu, z_logvar)

      

        # Concat vectors and decode
        zy = torch.cat([z, y], dim=1)
        x_mu, x_logvar = self.zy_x(zy)
        x_gen = self.reparam(x_mu, x_logvar)
        
        return x_gen, y



#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================