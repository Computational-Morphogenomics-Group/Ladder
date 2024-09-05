import wh.models as models
import wh.scripts as scripts # import train_pyro, train_pyro_disjoint_param, get_device
import wh.data.real_data as utils #import construct_labels, distrib_dataset
import anndata as ad
import scanpy as sc
from typing import Literal
import sys, warnings




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
                  batch_key : str = None):
        """
        Creates the required data objects to run the model       
        """
        try:
            # Set params
            self.label_style = 'concat' if self.model_type != 'SCANVI' else 'one-hot'
            self.factors = factors
            self.len_attrs = [len(pd.get_dummies(metadata[factor]).columns) for factor in factors]
            self.batch_key = batch_key
    

            if self.verbose: print(f'\nCondition classes : {factors}\nNumber of attributes per class : {len_attrs}')

            
            # Handle batch & create the datasets
            if self.batch_key is not None:
                self.dataset, self.levels, self.converter, self.batch_mapping = utils.construct_labels(self.anndat.X, self.anndat.obs, self.factors, style=self.label_style, batch_key=self.batch_key)
                self.train_set, self.test_set, self.train_loader, self.test_loader, self.l_mean, self.l_scale = utils.distrib_dataset(dataset, 
                                                                 levels, 
                                                                 batch_size=batch_size,
                                                                 batch_key=self.batch_key)

                self.batch_correction = True

            # If no batch
            else:
                self.dataset, self.levels, self.converter = utils.construct_labels(self.anndat.X, self.anndat.obs, self.factors, style=self.label_style)
                self.train_set, self.test_set, self.train_loader, self.test_loader, self.l_mean, self.l_scale = utils.distrib_dataset(dataset, 
                                                                 levels, 
                                                                 batch_size=batch_size,
                                                                 batch_key=self.batch_key)

                self.batch_correction = False
        

        # Not supposed to access this from outside normally
        except AttributeError :
            sys.exit("ERROR: There seems to be no model registered to the workflow.")
            

    
    def prep_model(self, 
                   factors : list,
                   batch_key : str = None,
                   model_type : Literal["SCVI", "SCANVI", "Patches"] = "Patches",
                   model_args : dict = None, 
                   optim_args : dict = None):

        """
        Creates the model object to be run. 
        """

        # Register model type
        self.model_type = model_type
        self.reconstruction = "ZINB" if self.config == cross_condition else "ZINB_LD"

        # Prepare the data
        self._prep_data(factors, batch_key)

        # Grab model constructor
        constructor = getattr(models.scvi_variants, self.model_type)
        try:
            self.model = constructor(self.anndata.X.shape[-1], self.l_mean, self.l_scale, batch_correction = self.batch_correction, reconstruction=self.reconstruction, **model_args)

        except Exception as e:
            warnings.warn("WARNING: model_args were not parsed correctly, returning to model defaults...")

        if self.verbose: print(f'Initialized {self.model_type} model.')
        
        # Fill in optimizer gaps
        for key in OPT_DEFAULTS.keys():
            if key not in optim_args.keys():
                optim_args[key] = OPT_DEFAULTS[key]

        # Grab model optimizer
        match self.model_type:
            case self.model_type if self.model_type in OPT_CLASS1 :
                self.optim_args = {
                    'optimizer': opt.Adam, 
                    'optim_args': {'lr': optim_args['lr'], 'eps' : optim_args['eps'], 'betas' : optim_args['betas']}, 
                    'gamma': optim_args['gamma'], 
                    'milestones': optim_args['milestones']}

            
            case self.model_type if self.model_type in OPT_CLASS2 :
                self.optim_args = {
                    'lr': optim_args['lr'], 'eps' : optim_args['eps'], 'betas' : optim_args['betas']
                }

        if self.verbose: print(f'Optimizer args parsed successfully.')

            
        
        


    
        
        
        
        
    