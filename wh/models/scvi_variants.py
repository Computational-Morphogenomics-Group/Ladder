import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from typing import Literal

from .basics import _broadcast_inputs, _make_func, _split_in_half
from torch.nn.functional import softplus, softmax
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
                 latent_dim=10, scale_factor=1.0, batch_correction=False, reconstruction : Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"] = "ZINB"):
         

        # Init params & hyperparams
        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.l_loc = l_loc
        self.l_scale = l_scale
        self.batch_correction = batch_correction         # Assume that batch is appended to input & latent if batch correction is applied
        self.reconstruction = reconstruction



        super(SCVI, self).__init__()

        # Setup NN functions
        match self.reconstruction:
            case "ZINB":
                self.x_decoder = _make_func(in_dims=self.latent_dim + int(self.batch_correction), hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="zinb")

            case "Normal":
                self.x_decoder = _make_func(in_dims=self.latent_dim + int(self.batch_correction), hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="normal")
            
            case "ZINB_LD" | "Normal_LD":
                self.x_decoder = nn.Linear(self.latent_dim + int(self.batch_correction), self.num_genes*2, bias=False)


        
        self.zl_encoder = _make_func(in_dims=self.num_genes + int(self.batch_correction), hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="+lognormal", dist_config="+lognormal")
        

        self.epsilon = 0.006

    
    # Model
    def model(self, x, y=None):
        pyro.module("scvi", self)

        # Inverse dispersions
        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes), constraint=constraints.positive)

        # Loop for mini-batch
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample("z", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))

            # If batch correction, pick corresponding loc scale
            if self.batch_correction:
                l_loc, l_scale = self.l_loc[x[..., -1].type(torch.int)], self.l_scale

            # Single size factor
            else :
                l_loc, l_scale = self.l_loc * x.new_ones(1), self.l_scale * x.new_ones(1)

            
            l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))


            # If batch corrected, use batch to go back. Else skip
            if self.batch_correction:
                z = torch.cat([z, x[..., -1].view(-1,1)], dim=-1)


            match self.reconstruction:
                
                case "ZINB":
                    gate_logits, mu = self.x_decoder(z)
                    nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

                case "Normal":
                    x_loc, x_scale = self.x_decoder(z)
                    x_dist = dist.Normal(x_loc, x_scale)

                case "ZINB_LD":
                    gate_logits, mu = _split_in_half(self.x_decoder(z))
                    mu = softmax(mu, dim=-1)
                    nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

                case "Normal_LD":
                    _z = z.reshape(-1, z.size(-1))
                    out = self.x_decoder(_z)
                    out = out.reshape(z.shape[:-1] + out.shape[-1:])
                
                    x_loc, x_scale = _split_in_half(out)
                    x_scale = softplus(x_scale)
                    x_dist = dist.Normal(x_loc, x_scale)
            

            # If batch corrected, we expect last index to be batch
            if self.batch_correction:
                pyro.sample("x", x_dist.to_event(1), obs=x[..., :-1])
            else:
                pyro.sample("x", x_dist.to_event(1), obs=x)

    
    # Guide
    def guide(self, x, y=None):
        pyro.module("scvi", self)
        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):

            # If batch corrected, this is expression appended with batch
            z_loc, z_scale, l_loc, l_scale = self.zl_encoder(x)
            
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    
    # Generate
    def generate(self, x, y_source=None, y_target=None):
        pyro.module("scvi", self)
        
        ## Encode
        z_loc, z_scale, l_loc, l_scale = self.zl_encoder(x)
            
        l_enc = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
        z_enc = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        # If batch correction, then append batch to latent
        if self.batch_correction:
            z_enc = torch.cat([z_enc, x[..., -1].view(-1,1)], dim=-1)

        match self.reconstruction:
        
            case "ZINB":
                gate_logits, mu = self.x_decoder(z_enc)
                nb_logits = (l_enc * mu + self.epsilon).log() - (theta.to(mu.device) + self.epsilon).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

            case "Normal":
                x_loc, x_scale = self.x_decoder(z_enc)
                x_dist = dist.Normal(x_loc, x_scale)

            
            case "ZINB_LD":
                gate_logits, mu = _split_in_half(self.x_decoder(z_enc))
                mu = softmax(mu, dim=-1)
                nb_logits = (l_enc * mu + self.epsilon).log() - (theta.to(mu.device) + self.epsilon).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

            case "Normal_LD":
                _z_enc = z_enc.reshape(-1, z_enc.size(-1))
                out = self.x_decoder(_z_enc)
                out = out.reshape(z_enc.shape[:-1] + out.shape[-1:])
                
                x_loc, x_scale = _split_in_half(out)
                x_scale = softplus(x_scale)
                x_dist = dist.Normal(x_loc, x_scale)
            
            
            
        x_rec = pyro.sample("x", x_dist.to_event(1))
        return x_rec

    # Save self
    def save(self, path="scvi_params"):
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")


    # Load
    def load(self, path="scvi_params", map_location=None):
        pyro.clear_param_store()

        if map_location is None:
            self.load_state_dict(torch.load(path + "_torch.pth"))
            pyro.get_param_store().load(path + "_pyro.pth")

        else:
            self.load_state_dict(torch.load(path + "_torch.pth", map_location=map_location))
            pyro.get_param_store().load(path + "_pyro.pth", map_location=map_location)


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
                 latent_dim=10, alpha=0.1, scale_factor=1.0, batch_correction=False, reconstruction : Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"] = "ZINB"):
         

        # Init params & hyperparams
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.num_genes = num_genes
        self.num_labels = num_labels
        self.latent_dim = latent_dim
        self.l_loc = l_loc
        self.l_scale = l_scale
        self.batch_correction = batch_correction         # Assume that batch is appended to input & latent if batch correction is applied
        self.reconstruction = reconstruction



        super(SCANVI, self).__init__()

        # Setup NN functions
        
        match self.reconstruction:
            case "ZINB":
                self.z2_decoder = _make_func(in_dims=self.latent_dim + self.num_labels, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
                
                self.x_decoder = _make_func(in_dims=self.latent_dim + int(self.batch_correction), hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="zinb")

            case "Normal":
                self.z2_decoder = _make_func(in_dims=self.latent_dim + self.num_labels, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
                
                self.x_decoder = _make_func(in_dims=self.latent_dim + int(self.batch_correction), hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="normal")

            case "ZINB_LD" | "Normal_LD":
                self.x_decoder = nn.Linear(self.latent_dim + self.num_labels, self.num_genes*2, bias=False)

                
                
        self.z2l_encoder = _make_func(in_dims=self.num_genes, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="+lognormal", dist_config="+lognormal")
        
        self.classifier = _make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_labels, last_config="default", dist_config="classifier")
        
        self.z1_encoder = _make_func(in_dims=self.num_labels + self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
        

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
            
            
            
            l_loc, l_scale = self.l_loc * x.new_ones(1), self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            match self.reconstruction:
                case "ZINB":
                    z2_loc, z2_scale = self.z2_decoder(z1_y)
                    z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))
                    
                    
                    if self.batch_correction:
                        z2 = torch.cat([z2, x[..., -1].view(-1,1)], dim=-1)
                    
                    gate_logits, mu = self.x_decoder(z2)
                    nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

                case "Normal":
                    z2_loc, z2_scale = self.z2_decoder(z1_y)
                    z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))
                    
                    
                    if self.batch_correction:
                        z2 = torch.cat([z2, x[..., -1].view(-1,1)], dim=-1)

                    x_loc, x_scale = self.x_decoder(z2)
                    x_dist = dist.Normal(x_loc, x_scale)

                
                case "ZINB_LD":
                    gate_logits, mu = _split_in_half(self.x_decoder(z1_y))
                    mu = softmax(mu, dim=-1)
                    nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)


                case "Normal_LD":
                    _z1_y = z1_y.reshape(-1, z1_y.size(-1))
                    out = self.x_decoder(_z1_y)
                    out = out.reshape(z1_y.shape[:-1] + out.shape[-1:])
                
                    x_loc, x_scale = _split_in_half(out)
                    x_scale = softplus(x_scale)
                    x_dist = dist.Normal(x_loc, x_scale)
                    
            
            
            pyro.sample("x", x_dist.to_event(1), obs=x)

    
    # Guide
    def guide(self, x, y):
        pyro.module("scanvi", self)
        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            
            z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            
            if self.reconstruction in ["ZINB_LD", "Normal_LD"]:
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1), infer={'is_auxiliary': True})
            
            else:
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))

            
            y_logits = self.classifier(z2)
            y_dist = dist.OneHotCategorical(logits=y_logits)
            classification_loss = y_dist.log_prob(y)
            pyro.factor("classification_loss", -self.alpha * classification_loss, has_rsample=False)

            z2_y = _broadcast_inputs([z2, y])
            z2_y = torch.cat(z2_y, dim=-1)
            z1_loc, z1_scale = self.z1_encoder(z2_y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


    # Function to move points between conditions
    @torch.no_grad()
    def generate(self, x, y_source=None, y_target=None):
        pyro.module("scanvi", self)
  
        ## Encode
        #Variational for rho & l
        z2_loc, z2_scale, l_loc, l_scale = self.z2l_encoder(x)
            
        l_enc = pyro.sample("l_enc", dist.LogNormal(l_loc, l_scale).to_event(1))
        z2_enc = pyro.sample("z2_enc", dist.Normal(z2_loc, z2_scale).to_event(1))


        # Variational for z
        z2_y = _broadcast_inputs([z2_enc, y_source])
        z2_y = torch.cat(z2_y, dim=-1)
        z1_loc, z1_scale = self.z1_encoder(z2_y)
        z1_enc = pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))

        
        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        z1_y = torch.cat([z1_enc, y_target], dim=-1)

        
        match self.reconstruction:
            case "ZINB":
                z2_loc, z2_scale = self.z2_decoder(z1_y)
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))
                    
                    
                if self.batch_correction:
                    z2 = torch.cat([z2, x[..., -1].view(-1,1)], dim=-1)
                    
                gate_logits, mu = self.x_decoder(z2)
                nb_logits = (l_enc * mu + self.epsilon).log() - (theta.to(mu.device) + self.epsilon).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

            case "Normal":
                z2_loc, z2_scale = self.z2_decoder(z1_y)
                z2 = pyro.sample("z2", dist.Normal(z2_loc, z2_scale).to_event(1))
                    
                    
                if self.batch_correction:
                    z2 = torch.cat([z2, x[..., -1].view(-1,1)], dim=-1)

                x_loc, x_scale = self.x_decoder(z2)
                x_dist = dist.Normal(x_loc, x_scale)

                
            case "ZINB_LD":
                gate_logits, mu = _split_in_half(self.x_decoder(z1_y))
                mu = softmax(mu, dim=-1)
                nb_logits = (l_enc * mu + self.epsilon).log() - (theta.to(mu.device) + self.epsilon).log()
                x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)


            case "Normal_LD":
                _z1_y = z1_y.reshape(-1, z1_y.size(-1))
                out = self.x_decoder(_z1_y)
                out = out.reshape(z1_y.shape[:-1] + out.shape[-1:])
                
                x_loc, x_scale = _split_in_half(out)
                x_scale = softplus(x_scale)
                x_dist = dist.Normal(x_loc, x_scale)


            
        #Observe the datapoint x using the observation distribution x_dist
        x_rec = pyro.sample("x", x_dist.to_event(1))

        return x_rec


    # Save self
    def save(self, path="scanvi_params"):
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")


    # Load
    def load(self, path="scanvi_params", map_location=None):
        pyro.clear_param_store()

        if map_location is None:
            self.load_state_dict(torch.load(path + "_torch.pth"))
            pyro.get_param_store().load(path + "_pyro.pth")

        else:
            self.load_state_dict(torch.load(path + "_torch.pth", map_location=map_location))
            pyro.get_param_store().load(path + "_pyro.pth", map_location=map_location)



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
                 latent_dim=10, num_layers=1, hidden_dim=128, alphas=[0.1, 1], scale_factor=1.0, batch_correction=False, ld_sparsity=0, ld_normalize=False, reconstruction : Literal["ZINB", "Normal", "ZINB_LD", "Normal_LD"] = "ZINB"):

        
        # Init params & hyperparams
        self.alphas = alphas
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
        self.batch_correction = batch_correction         # Assume that batch is appended to input & latent if batch correction is applied
        self.reconstruction = reconstruction   # Distribution for the reconstruction
        self.sparsity = ld_sparsity  # Sparsity, used only with LD
        self.normalize = ld_normalize # Normalization, adds bias to LD
        
        super(CSSCVI, self).__init__()

        
        # Setup NN functions
       

        match self.reconstruction:
            
            case "ZINB":
                self.rho_decoder = _make_func(in_dims=self.latent_dim + (self.w_dim * self.num_labels), hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
                
                self.x_decoder = _make_func(in_dims=self.latent_dim + int(self.batch_correction), hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="zinb")

            case "Normal":
                self.rho_decoder = _make_func(in_dims=self.latent_dim + (self.w_dim * self.num_labels), hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
                
                self.x_decoder = _make_func(in_dims=self.latent_dim + int(self.batch_correction), hidden_dims=[hidden_dim]*num_layers, out_dim=self.num_genes, last_config="reparam", dist_config="normal")

            case "ZINB_LD" | "Normal_LD":
                self.x_decoder = nn.Linear(self.latent_dim + (self.w_dim * self.num_labels), self.num_genes*2, bias=self.normalize)
        
        
        self.rho_l_encoder = _make_func(in_dims=self.num_genes, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="+lognormal", dist_config="+lognormal")



        for i in range(len(self.len_attrs)):
            setattr(self, f"classifier_z_y{i}",  _make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.len_attrs[i], last_config="default", dist_config="classifier"))
        
        
        self.z_encoder = _make_func(in_dims=self.latent_dim, hidden_dims=[hidden_dim]*num_layers, out_dim=self.latent_dim, last_config="reparam", dist_config="normal")
        self.w_encoder = _make_func(in_dims=self.latent_dim + self.num_labels, hidden_dims=[hidden_dim]*num_layers, out_dim=self.w_dim*self.num_labels, last_config="reparam", dist_config="normal")

        self.epsilon = 0.006

    
    # Model
    def model(self, x, y):
        pyro.module("csscvi", self)

        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes),
                           constraint=constraints.positive)


        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample("z", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))


            # Keep tracked attributes in a list
            y_s = []
            attr_track = 0
            
            for i in pyro.plate("len_attrs", len(self.len_attrs)):
                next_track = attr_track + self.len_attrs[i]
                y_attr = pyro.sample(f"y_{i}", dist.OneHotCategorical(logits=x.new_zeros(self.len_attrs[i])), obs=y[..., attr_track : next_track])
                y_s.append(y_attr)
                
                attr_track = next_track
                        

            w_loc = torch.concat([self.concat_lat_dims(y, self.w_locs, self.w_dim) for y in y_s], dim = -1)
            w_scale = torch.concat([self.concat_lat_dims(y, self.w_scales, self.w_dim) for y in y_s], dim = -1)
           
            
            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

            zw = torch.cat([z, w], dim=-1)

            l_loc, l_scale = self.l_loc * x.new_ones(1), self.l_scale * x.new_ones(1)
            l = pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))


            # Part to modify if changing the decoder
            match self.reconstruction:
                case "ZINB":
                    rho_loc, rho_scale = self.rho_decoder(zw)
                    rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

            
                    if self.batch_correction:
                        rho = torch.cat([rho, x[..., -1].view(-1,1)], dim=-1)
                    
                    gate_logits, mu = self.x_decoder(rho)
                    nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

                case "Normal":
                    rho_loc, rho_scale = self.rho_decoder(zw)
                    rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))
    
            
                    if self.batch_correction:
                        rho = torch.cat([rho, x[..., -1].view(-1,1)], dim=-1)
                    
                    x_loc, x_scale = self.x_decoder(rho)
                    x_dist = dist.Normal(x_loc, x_scale)

                case "ZINB_LD":
                    gate_logits, mu = _split_in_half(self.x_decoder(zw))
                    mu = softmax(mu, dim=-1)
                    nb_logits = (l * mu + self.epsilon).log() - (theta + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)


                case "Normal_LD":
                    _zw = zw.reshape(-1, zw.size(-1))
                    out = self.x_decoder(_zw)
                    out = out.reshape(zw.shape[:-1] + out.shape[-1:])
                
                    x_loc, x_scale = _split_in_half(out)
                    x_scale = softplus(x_scale)
                    x_dist = dist.Normal(x_loc, x_scale)

                    

            pyro.sample("x", x_dist.to_event(1), obs=x)

    
    # Guide
    def guide(self, x, y):
        pyro.module("csscvi", self)
        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            
            # Variational for rho & l
            rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)
            
            
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))

            if self.reconstruction in ["ZINB_LD", "Normal_LD"]:
                rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1), infer={'is_auxiliary': True})
            
            else:
                rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))


            # Variational for w & z
            rho_y = _broadcast_inputs([rho, y])
            rho_y = torch.cat(rho_y, dim=-1)
            
            w_loc, w_scale = self.w_encoder(rho_y)
            z_loc, z_scale = self.z_encoder(rho)

            w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        
            # Classification for w (good) and z (bad)

            # Keep track over list
            classification_loss_z = 0
            attr_track = 0
            
            for i in pyro.plate("len_attrs", len(self.len_attrs)):
                next_track = attr_track + self.len_attrs[i]
                
                cur_func = getattr(self, f"classifier_z_y{i}")
                cur_logits = cur_func(z)
                cur_dist =  dist.OneHotCategorical(logits=cur_logits)
                classification_loss_z += self.alphas[i] * cur_dist.log_prob(y[..., attr_track : next_track])

                attr_track = next_track
            
                                        
            pyro.factor("classification_loss", classification_loss_z, has_rsample=False) # Want this maximized so positive sign in guide

            if (self.reconstruction in ["ZINB_LD", "Normal_LD"]) and self.sparsity:
                params = list(self.x_decoder.parameters())[0].T[self.latent_dim:].clone() 
                _, x_loc_params = params.reshape(params.shape[:-1] + (2, -1)).unbind(-2)
                pyro.factor("l1_loss", x_loc_params.sum().abs(), has_rsample=False) # sparsity

    # Adverserial
    def adverserial(self, x, y):
        pyro.module("csscvi", self)
        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # Variational for rho & l
            rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)
            
            pyro.sample("l", dist.LogNormal(l_loc, l_scale).to_event(1))
            rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

            # Variational for w & z
            rho_y = _broadcast_inputs([rho, y])
            rho_y = torch.cat(rho_y, dim=-1)
            
            z_loc, z_scale = self.z_encoder(rho)

            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


            # Classification for w (good) and z (bad)

            # Keep track over list
            classification_loss_z = 0
            attr_track = 0
            
            for i in pyro.plate("len_atrrs", len(self.len_attrs)):
                next_track = attr_track + self.len_attrs[i]
                
                cur_func = getattr(self, f"classifier_z_y{i}")
                cur_logits = cur_func(z)
                cur_dist =  dist.OneHotCategorical(logits=cur_logits)
                classification_loss_z += self.alphas[i] * cur_dist.log_prob(y[..., attr_track : next_track])

                attr_track = next_track
            
            
                                
            
            return -1.0*classification_loss_z
        
    
    # Function to move points between conditions
    @torch.no_grad()
    def generate(self, x, y_source=None, y_target=None):
        pyro.module("csscvi", self)
  
        ## Encode
        #Variational for rho & l
        rho_loc, rho_scale, l_loc, l_scale = self.rho_l_encoder(x)
            
        rho_enc = pyro.sample("rho_enc", dist.Normal(rho_loc, rho_scale).to_event(1))
        l_enc = pyro.sample("l_enc", dist.LogNormal(l_loc, l_scale).to_event(1))


        # Variational for w & z
        ## TODO: Search the attribute space instead of picking a single sample

        # Keep tracked attributes in a list
        y_s = []
        attr_track = 0
            
        for i in range(len(self.len_attrs)):
            next_track = attr_track + self.len_attrs[i]
            y_s.append(y_target[..., attr_track : next_track])
                
            attr_track = next_track


        w_loc = torch.concat([self.concat_lat_dims(y, self.w_locs, self.w_dim) for y in y_s], dim = -1)
        w_scale = torch.concat([self.concat_lat_dims(y, self.w_scales, self.w_dim) for y in y_s], dim = -1)
           
        z_loc, z_scale = self.z_encoder(rho_enc)
            
        w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))
        z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

        
        ## Decode
        theta = dict(pyro.get_param_store())["inverse_dispersion"].detach()

        zw = torch.cat([z, w], dim=-1)

        match self.reconstruction:
                case "ZINB":
                    rho_loc, rho_scale = self.rho_decoder(zw)
                    rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))

            
                    if self.batch_correction:
                        rho = torch.cat([rho, x[..., -1].view(-1,1)], dim=-1)

                    gate_logits, mu = self.x_decoder(rho)

                    nb_logits = (l_enc * mu + self.epsilon).log() - (theta.to(mu.device) + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)

                case "Normal":
                    rho_loc, rho_scale = self.rho_decoder(zw)
                    rho = pyro.sample("rho", dist.Normal(rho_loc, rho_scale).to_event(1))
    
            
                    if self.batch_correction:
                        rho = torch.cat([rho, x[..., -1].view(-1,1)], dim=-1)
                    
                    x_loc, x_scale = self.x_decoder(rho)
                    x_dist = dist.Normal(x_loc, x_scale)

                case "ZINB_LD":
                    gate_logits, mu = _split_in_half(self.x_decoder(zw))
                    mu = softmax(mu, dim=-1)
                    nb_logits = (l_enc * mu + self.epsilon).log() - (theta.to(mu.device) + self.epsilon).log()
                    x_dist = dist.ZeroInflatedNegativeBinomial(gate_logits=gate_logits, total_count=theta, logits=nb_logits, validate_args=False)


                case "Normal_LD":
                    _zw = zw.reshape(-1, zw.size(-1))
                    out = self.x_decoder(_zw)
                    out = out.reshape(zw.shape[:-1] + out.shape[-1:])
                
                    x_loc, x_scale = _split_in_half(out)
                    x_scale = softplus(x_scale)
                    x_dist = dist.Normal(x_loc, x_scale)
            
        #Observe the datapoint x using the observation distribution x_dist
        x_rec = pyro.sample("x", x_dist.to_event(1))

        return x_rec


    # Save self
    def save(self, path="csscvi_params"):
        torch.save(self.state_dict(), path + "_torch.pth")
        pyro.get_param_store().save(path + "_pyro.pth")


    # Load
    def load(self, path="csscvi_params", map_location=None):
        pyro.clear_param_store()

        if map_location is None:
            self.load_state_dict(torch.load(path + "_torch.pth"))
            pyro.get_param_store().load(path + "_pyro.pth")

        else:
            self.load_state_dict(torch.load(path + "_torch.pth", map_location=map_location))
            pyro.get_param_store().load(path + "_pyro.pth", map_location=map_location)



            
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
