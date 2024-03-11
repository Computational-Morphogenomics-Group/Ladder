import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape

## Helpers and scANVI taken from https://pyro.ai/examples/scanvi.html
# Splits a tensor in half along the final dimension
def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)

# Helper for broadcasting inputs to neural net
def broadcast_inputs(input_args):
    shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
    input_args = [s.expand(shape) for s in input_args]
    return input_args

# FC layer maker 
def make_fc(dims):
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])  # Exclude final ReLU non-linearity
##


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================


class SCANVI(nn.Module):
    """SCANVI"""
    def __init__(self, num_genes, num_labels, l_loc, l_scale,
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
        self.z2_decoder = Z2Decoder(z1_dim=self.latent_dim, y_dim=self.num_labels,
                                    z2_dim=self.latent_dim, hidden_dims=[50])
        self.x_decoder = XDecoder(num_genes=num_genes, hidden_dims=[100], z2_dim=self.latent_dim)
        self.z2l_encoder = Z2LEncoder(num_genes=num_genes, z2_dim=self.latent_dim, hidden_dims=[100])
        self.classifier = Classifier(z2_dim=self.latent_dim, hidden_dims=[50], num_labels=num_labels)
        self.z1_encoder = Z1Encoder(num_labels=num_labels, z1_dim=self.latent_dim,
                                    z2_dim=self.latent_dim, hidden_dims=[50])

        self.epsilon = 0.006

    # Model
    def model(self, x, y):
        pyro.module("scanvi", self)

        # Inverse dispersions
        theta = pyro.param("inverse_dispersion", 10.0 * x.new_ones(self.num_genes), constraint=constraints.positive)

        
        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z1 = pyro.sample("z1", dist.Normal(0, x.new_ones(self.latent_dim)).to_event(1))
            y = pyro.sample("y", dist.OneHotCategorical(logits=x.new_zeros(self.num_labels)), obs=y)

            
            z2_loc, z2_scale = self.z2_decoder(z1, y)
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

            
            z1_loc, z1_scale = self.z1_encoder(z2, y)
            pyro.sample("z1", dist.Normal(z1_loc, z1_scale).to_event(1))


#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================
#================================================================================================