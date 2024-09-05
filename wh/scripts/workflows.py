import wh.models as models
import wh.scripts as scripts # import train_pyro, train_pyro_disjoint_param, get_device
import wh.data.real_data as utils #import construct_labels, distrib_dataset
import anndata as ad
import scanpy as sc
from typing import Literal
import sys, warnings
import pandas as pd 
import math
import torch.optim as opt



warnings.simplefilter('always', UserWarning)


class BaseWorkflow:
    """
    Implements the base workflow for running any of the pipelines.
    """

    # Static variable for optimizer choice
    OPT_CLASS1 = ['SCVI', 'SCANVI'] # These do not require a complex optimizer loop
    OPT_CLASS2 = ['Patches'] # These require adversarial optimizer for the latent

    # Static lookup for optimizer defaults
    OPT_DEFAULTS = {
        'lr' : 1e-2, 
        'eps' : 1e-2,
        'betas' : (0.90, 0.999),
        'gamma': 1,
        'milestones': [1e10]
    }

    # Static list of keys allowed in optim args
    OPT_LIST = ['optimizer', 'optim_args', 'gamma', 'milestones', 'lr', 'eps', 'betas']

    # Constructor
    def __init__(self, 
                 anndata : ad.AnnData, 
                 config : Literal["cross-condition", "interpretable"] = "cross-condition",
                 verbose : bool = False):

        # Set internals
        self.anndata = anndata
        self.config = config
        self.verbose = verbose
        self.batch_key = None
        self.levels = None
        self.model_type = None


        if self.verbose: print(f'Initialized workflow to run {config} model.')

    
    def __str__(self):
        return f"""
{self.__class__.__name__} with parameters:
=================================================
Config: {self.config}
Verbose: {self.verbose}
Levels: {self.levels}
Batch Key: {self.batch_key}
Model: {self.model_type}
"""
    
    def __repr__(self):
        return f'<Workflow / Config: {self.config}, Verbose: {self.verbose}, Levels: {self.levels}, Batch Key: {self.batch_key}, Model: {self.model_type}>'

    
        

    # Data prep - private call
    def _prep_data(self, 
                  factors : list, 
                  batch_key : str = None,
                  minibatch_size : int = 128):
        """
        Creates the required data objects to run the model       
        """

        if self.model_type is not None: # Flag cleared, proceed to data setup
            self.label_style = 'concat' if self.model_type != 'SCANVI' else 'one-hot'
            self.factors = factors
            self.len_attrs = [len(pd.get_dummies(self.anndata.obs[factor]).columns) for factor in factors]
            self.batch_key = batch_key
            self.minibatch_size = minibatch_size
    

            if self.verbose: print(f'\nCondition classes : {self.factors}\nNumber of attributes per class : {self.len_attrs}')

            
            # Handle batch & create the datasets
            if self.batch_key is not None:
                self.dataset, self.levels, self.converter, self.batch_mapping = utils.construct_labels(self.anndata.X, self.anndata.obs, self.factors, style=self.label_style, batch_key=self.batch_key)
                self.train_set, self.test_set, self.train_loader, self.test_loader, self.l_mean, self.l_scale = utils.distrib_dataset(self.dataset, 
                                                                 self.levels, 
                                                                 batch_size=128,
                                                                 batch_key=self.batch_key)

                self.batch_correction = True

            # If no batch
            else:
                self.dataset, self.levels, self.converter = utils.construct_labels(self.anndata.X, self.anndata.obs, self.factors, style=self.label_style)
                self.train_set, self.test_set, self.train_loader, self.test_loader, self.l_mean, self.l_scale = utils.distrib_dataset(self.dataset, 
                                                                 self.levels, 
                                                                 batch_size=128,
                                                                 batch_key=self.batch_key)

                self.batch_correction = False
        

        else:
            warnings.warn("ERROR: There seems to be no model registered to the workflow. Make sure not to run this function directly if you did so. You must instead run the 'prep_model()' function.")

    
    def _fetch_model_args(self, model_args):
        """
        Used to modify model_args efficiently depending on the model used.
        """
        # For all models
        model_args["reconstruction"] = self.reconstruction
        model_args["batch_correction"] = self.batch_correction

        # Model specific
        match self.model_type:
            case "SCANVI":
                model_args["num_labels"] = math.prod(self.len_attrs)
         
            case "Patches":
                model_args["num_labels"] = sum(self.len_attrs)
                model_args["len_attrs"] = self.len_attrs
        
        return model_args

    
    def _clear_bad_optim_args(self):
        """
        Clears optim_args keys that won't be used downstream.
        """
        self.optim_args = {k : v for k,v in self.optim_args.items() if k in self.OPT_LIST}
            
    
    def prep_model(self, 
                   factors : list,
                   batch_key : str = None,
                   minibatch_size : int = 128,
                   model_type : Literal["SCVI", "SCANVI", "Patches"] = "Patches",
                   model_args : dict = None, 
                   optim_args : dict = None):

        """
        Creates the model object to be run. 
        """

        # Register model type
        self.model_type = model_type
        self.reconstruction = "ZINB" if self.config == 'cross_condition' else "ZINB_LD"

        # Prepare the data
        self._prep_data(factors, batch_key, minibatch_size)

        # Grab model constructor
        constructor = getattr(models.scvi_variants, self.model_type)
        
        ## Additional inputs for models
        
        if model_args is None: model_args = {}  ### Get one if not provided

        model_args = self._fetch_model_args(model_args)
      
        # Construct model
        try:
            self.model = constructor(self.anndata.X.shape[-1], self.l_mean, self.l_scale, **model_args)
            
        except Exception as e:
            warnings.warn("\nINFO: model_args ignored, using model defaults...")
            model_args = self._fetch_model_args({})
            self.model = constructor(self.anndata.X.shape[-1], self.l_mean, self.l_scale, **model_args)


        if self.verbose: print(f'\nInitialized {self.model_type} model.\nModel arguments: {model_args}')

        # Get optimizer args if not provided
        if optim_args is None: optim_args = {}
        
        # Fill in optimizer gaps
        for key in self.OPT_DEFAULTS.keys():
            if key not in optim_args.keys():
                optim_args[key] = self.OPT_DEFAULTS[key]

        # Grab model optimizer
        match self.model_type:
            case self.model_type if self.model_type in self.OPT_CLASS1 :
                self.optim_args = {
                    'optimizer': opt.Adam, 
                    'optim_args': {'lr': optim_args['lr'], 'eps' : optim_args['eps'], 'betas' : optim_args['betas']}, 
                    'gamma': optim_args['gamma'], 
                    'milestones': optim_args['milestones']
                }

            
            case self.model_type if self.model_type in self.OPT_CLASS2 :
                self.optim_args = {
                    'lr': optim_args['lr'], 'eps' : optim_args['eps'], 'betas' : optim_args['betas']
                }

        # Clear whatever is unused
        self._clear_bad_optim_args()

        if self.verbose: print(f'\nOptimizer args parsed successfully. Final arguments: {self.optim_args}')

            
        
        


    
        
        
        
        
    