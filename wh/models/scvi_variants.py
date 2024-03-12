import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from.basics import broadcast_inputs, make_func
import numpy as np

#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================


class SCVI(nn.Module):
    """
    SCVI
    """
    
    def __init__(self, num_genes, l_loc, l_scale, hidden_dim=128, num_layers=1,
                 latent_dim=10, scale_factor=1.0):
         

        # Init params & hyperparams
        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.l_loc = l_loc
        self.l_scale = l_scale



        super(SCVI, self).__init__()

        # Setup NN functions
        self.z_decoder = make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
        self.x_decoder = make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="zinb")
        self.zl_encoder = make_func(in_dims=self.num_genes, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="+lognormal", dist_config="+lognormal")
        

        self.epsilon = 0.006

    # Model
    def model(self, x, y=None):
        pyro.module("scvi", self)

        # Inverse dispersions
        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes), constraint=constraints.positive)

        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample("z", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            
            l_loc, l_scale = self.l_loc * x.new_ones(1), self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            
            gate_logits, mu = self.x_decoder(z)
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta,
                                                       logits=nb_logits, validate_args=False)
            
            
            pyro.sample("x", x_dist.to_event(1), obs=x)

    
    # Guide
    def guide(self, x, y=None):
        pyro.module("scvi", self)
        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            
            z_loc, z_scale, l_loc, l_scale = self.zl_encoder(x)
            
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


    # Save self
    def save(self, path="scvi_params"):
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")


    # Load
    def load(self, path="scvi_params"):
        pyro.clear_param_store()
        self.load_state_dict(torch.load(path + "_torch.pth"))
        pyro.get_param_store().load(path + "_pyro.pth")


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================

## scANVI taken from https://pyro.ai/examples/scanvi.html
class SCANVI(nn.Module):
    """
    SCANVI
    """
    
    def __init__(self, num_genes, num_labels, l_loc, l_scale, hidden_dim=128, num_layers=1,
                 latent_dim=10, alpha=0.01, scale_factor=1.0):
         

        # Init params & hyperparams
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.l_loc = l_loc
        self.l_scale = l_scale



        super(SCANVI, self).__init__()

        # Setup NN functions
        self.z2_decoder = make_func(in_dims=self.latent_dim + self.num_labels, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
        self.x_decoder = make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="zinb")
        self.z2l_encoder = make_func(in_dims=self.num_genes, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="+lognormal", dist_config="+lognormal")
        self.classifier = make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_labels, last_config="default", dist_config="classifier")
        self.z1_encoder = make_func(in_dims=self.num_labels + self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
        

        self.epsilon = 0.006

    # Model
    def model(self, x, y):
        pyro.module("scanvi", self)

        # Inverse dispersions
        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes), constraint=constraints.positive)

        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample("z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            y = pyro.sample("y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y)

            z1_y = torch.cat([z1, y], dim=-1)
            z2_loc, z2_scale = self.z2_decoder(z1_y)
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))
            
            
            l_loc, l_scale = self.l_loc * x.new_ones(1), self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            
            gate_logits, mu = self.x_decoder(z2)
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta,
                                                       logits=nb_logits, validate_args=False)
            
            
            pyro.sample("x", x_dist.to_event(1), obs=x)

    
    # Guide
    def guide(self, x, y):
        pyro.module("scanvi", self)
        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            
            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            classification_loss = y_dist.log_prob(y)
            pyro.factor("classification_loss", -self.alpha * classification_loss, has_rsample=False)

            z2_y = broadcast_inputs([z2, y])
            z2_y = torch.cat(z2_y, dim=-1)
            z1_loc, z1_scale = self.z1_encoder(z2_y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


    # Function to move points between conditions
    @torch.no_grad()
    def generate(self, x, y):
        pyro.module("scanvi", self)
  
        ## Encode
        #Variational for rho & l
        z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            
        l_enc = pyro.sample("l_enc", dist.LogNormal(l_loc, l_scale).to_event(1))
        z2_enc = pyro.sample("z2_enc", dist.Normal(z2_loc, z2_scale).to_event(1))


        # Variational for w & z
        z2_y = broadcast_inputs([z2, y])
        z2_y = torch.cat(z2_y, dim=-1)
        z1_loc, z1_scale = self.z1_encoder(z2_y)
        z1_enc = pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

        
        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        z1_y = torch.cat([z1_enc, y], dim=-1)
        z2_loc, z2_scale = self.z2_decoder(z1_y)
        z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))


        gate_logits, mu = self.x_decoder(z2)
        nb_logits = (l_enc * mu + self.epsilon).log() - (theta + self.epsilon).log()
        x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta,
                                                       logits=nb_logits, validate_args=False)
            
        #Observe the datapoint x using the observation distribution x_dist
        x_rec = pyro.sample("x", x_dist.to_event(1))

        return x_rec


    # Save self
    def save(self, path="scanvi_params"):
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")


    # Load
    def load(self, path="scanvi_params"):
        pyro.clear_param_store()
        self.load_state_dict(torch.load(path + "_torch.pth"))
        pyro.get_param_store().load(path + "_pyro.pth")



#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================



class CSSCVI(nn.Module):
    """
    CSSCVI
    """

    @staticmethod
    def concat_lat_dims(labels, ref_list, dim):
        """
        Function to organize W prior from labels.
        """
        idxs = labels.int()
        return torch.tensor(np.array([np.concatenate([[ref_list[num]]*dim for num in elem]) for elem in idxs])).type_as(labels).to(labels.device)

    
    def __init__(self, num_genes, num_labels, l_loc, l_scale, w_loc=[0,3], w_scale=[0.1,1], w_dim=10, len_attrs=[3,2],
                 latent_dim=10, num_layers=1, hidden_dim=128, alpha=0.01, scale_factor=1.0):

        
        # Init params & hyperparams
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.w_dim = w_dim # Latent dimension for each label
        self.l_loc = l_loc
        self.l_scale = l_scale
        self.w_locs = w_loc # Prior means for attribute being 0,1 (indices correspond to attribute value)
        self.w_scales = w_scale # Prior scales for attribute being 0,1 (indices correspond to attribute value)
        self.len_attrs=len_attrs # List keeping number of possibilities for each attribute

        super(CSSCVI, self).__init__()

        
        # Setup NN functions
        self.rho_decoder = make_func(in_dims=self.latent_dim + (self.w_dim * self.num_labels), hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
        self.x_decoder = make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="zinb")
        self.rho_l_encoder = make_func(in_dims=self.num_genes, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="+lognormal", dist_config="+lognormal")


        ## TODO: Keep these in a list to generalize over arbitrary number of attributes with different sizes
        self.classifier_w_y1 = make_func(in_dims=self.w_dim*self.num_labels, hidden_dims=[hidden_dim]*num_layers, out_dim=self.len_attrs[0], last_config="default", dist_config="classifier")
        self.classifier_w_y2 = make_func(in_dims=self.w_dim*self.num_labels, hidden_dims=[hidden_dim]*num_layers, out_dim=self.len_attrs[1], last_config="default", dist_config="classifier")
        
        self.z_encoder = make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
        self.w_encoder = make_func(in_dims=self.latent_dim + self.num_labels, hidden_dims=[hidden_dim]*num_layers, out_dim=self.w_dim*self.num_labels, last_config="reparam", dist_config="normal")

        self.epsilon = 0.006

    
    # Model
    def model(self, x, y):
        pyro.module("csscvi", self)

        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes),
                           constraint=constraints.positive)


        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample("z", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))


            ## TODO: Keep these in a list to generalize over arbitrary number of attributes with different sizes 
            y1 = pyro.sample("y1", dist.OneHotCategorical(logits=x.new_zeros(3)), obs=y[..., :self.len_attrs[0]])
            y2 = pyro.sample("y2", dist.OneHotCategorical(logits=x.new_zeros(2)), obs=y[..., self.len_attrs[0]:])            
            w_loc = torch.concat([self.concat_lat_dims(y1, self.w_locs, self.w_dim), self.concat_lat_dims(y2, self.w_locs, self.w_dim)], dim = -1)
            w_scale = torch.concat([self.concat_lat_dims(y1, self.w_scales, self.w_dim), self.concat_lat_dims(y2, self.w_scales, self.w_dim)], dim = -1)
            
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

            zw = torch.cat([z, w], dim=-1)
            rho_loc, rho_scale = self.rho_decoder(zw)
            rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))
            
            l_loc, l_scale = self.l_loc * x.new_ones(1), self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))


            gate_logits, mu = self.x_decoder(rho)
            nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta,
                                                       logits=nb_logits, validate_args=False)
            
            pyro.sample("x", x_dist.to_event(1), obs=x)

    
    # Guide
    def guide(self, x, y):
        pyro.module("csscvi", self)
        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # Variational for rho & l
            rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)
            
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

            # Variational for w & z
            rho_y = broadcast_inputs([rho, y])
            rho_y = torch.cat(rho_y, dim=-1)
            
            w_loc, w_scale = self.w_encoder(rho_y)
            z_loc, z_scale = self.z_encoder(rho)

            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


            # Classification for w (good) and z (bad)

            ## TODO: Keep these in a list to generalize over arbitrary number of attributes with different sizes
            w_y1_logits, w_y2_logits = self.classifier_w_y1(w), self.classifier_w_y2(w)
            w_y1_dist, w_y2_dist = dist.OneHotCategorical(logits=w_y1_logits), dist.OneHotCategorical(logits=w_y2_logits)

                
           
            classification_loss_w = w_y1_dist.log_prob(y[..., :self.len_attrs[0]]) + w_y2_dist.log_prob(y[..., self.len_attrs[0]:])
                
            pyro.factor("classification_loss", -self.alpha * classification_loss_w, has_rsample=False)


    
    # Function to move points between conditions
    @torch.no_grad()
    def generate(self, x, y):
        pyro.module("csscvi", self)
  
        ## Encode
        #Variational for rho & l
        rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)
            
        rho_enc = pyro.sample("rho_enc", dist.Normal(rho_loc, rho_scale).to_event(1))
        l_enc = pyro.sample("l_enc", dist.LogNormal(l_loc, l_scale).to_event(1))


        # Variational for w & z
        rho_enc_y = broadcast_inputs([rho_enc, y])
        rho_enc_y = torch.cat(rho_enc_y, dim=-1)
        
        w_loc, w_scale = self.w_encoder(rho_enc_y)
        z_loc, z_scale = self.z_encoder(rho_enc)

        w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        
        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        zw = torch.cat([z, w], dim=-1)
        rho_loc, rho_scale = self.rho_decoder(zw)
        rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))
            

        gate_logits, mu = self.x_decoder(rho)
        nb_logits = (l_enc * mu + self.epsilon).log() - (theta + self.epsilon).log()
        x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta,
                                                       logits=nb_logits, validate_args=False)
            
        #Observe the datapoint x using the observation distribution x_dist
        x_rec = pyro.sample("x", x_dist.to_event(1))

        return x_rec


    # Save self
    def save(self, path="csscvi_params"):
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")


    # Load
    def load(self, path="csscvi_params"):
        pyro.clear_param_store()
        self.load_state_dict(torch.load(path + "_torch.pth"))
        pyro.get_param_store().load(path + "_pyro.pth")


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================