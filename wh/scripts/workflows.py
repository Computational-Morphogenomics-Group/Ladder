import wh.models as models
import wh.scripts as scripts # import train_pyro, train_pyro_disjoint_param, get_device
import wh.data.real_data as utils #import construct_labels, distrib_dataset
import anndata as ad
import scanpy as sc
from typing import Literal
import sys, warnings
import pandas as pd 
import math
import pyro.optim as opt
import pyro, torch 
import numpy as np



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

    # Static list of registered metrics
    # Dict for pretty printing
    METRICS_REG = {'rmse' : 'RMSE', 'corr' : 'Profile Correlation', 'swd' : '2-Sliced Wasserstein', 'CD' : 'Chamfer Discrepancy'}



    
    # Constructor
    def __init__(self, 
                 anndata : ad.AnnData, 
                 config : Literal["cross-condition", "interpretable"] = "cross-condition",
                 verbose : bool = False,
                 random_seed : int = None):

        # Set internals
        self.anndata = anndata
        self.config = config
        self.verbose = verbose
        self.random_seed = random_seed
        self.batch_key = None
        self.levels = None
        self.model_type = None

        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            pyro.util.set_rng_seed(self.random_seed)

        if self.verbose: print(f'Initialized workflow to run {config} model.')
        
    
    def __str__(self):
        return f"""
{self.__class__.__name__} with parameters:
=================================================
Config: {self.config}
Verbose: {self.verbose}
Random Seed: {self.random_seed}
Levels: {self.levels}
Batch Key: {self.batch_key}
Model: {self.model_type}
"""
    
    def __repr__(self):
        return f'<Workflow / Config: {self.config}, Random Seed: {self.random_seed}, Verbose: {self.verbose}, Levels: {self.levels}, Batch Key: {self.batch_key}, Model: {self.model_type}>'

    
    



    
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
        model_args["scale_factor"] = 1./(self.minibatch_size * self.anndata.X.shape[-1])

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
            


    def _register_latent_dims(self):
        """
        Registers latent dimensions to be used downstream
        """
        match self.model_type:
            case self.model_type if self.model_type in self.OPT_CLASS1 :
                self.latent_dim = self.model.latent_dim

            
            case self.model_type if self.model_type in self.OPT_CLASS2 :
                self.latent_dim = self.model.latent_dim
                self.w_dim = self.model.w_dim
        

    
    
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

        # Flush params if needed
        pyro.clear_param_store()

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


        # Register latents to model
        self._register_latent_dims()


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

            
        
    def run_model(self, max_epochs : int = 300, convergence_threshold : float = 1e-3, classifier_warmup : int = 0, params_save_path : str = None):
        """
        Train the model. 
        """


        if dict(pyro.get_param_store()):
            warnings.warn("WARNING: Retraining without resetting parameters is discouraged. Please call prep_model() again if you wish to rerun training.")
        
        if self.verbose: print(f'Training initialized for a maximum of {max_epochs}, with convergence eps {convergence_threshold}.')
        if self.verbose and params_save_path is not None: print(f'Model parameters will be saved to path: {params_save_path} ')
        
        # Match the funtion to run
        match self.model_type:
            case self.model_type if self.model_type in self.OPT_CLASS1:
                self.model, self.train_loss, self.test_loss = scripts.training.train_pyro(self.model, train_loader=self.train_loader, test_loader=self.test_loader, verbose=True, num_epochs=max_epochs, convergence_threshold=convergence_threshold, optim_args = self.optim_args)

            case self.model_type if self.model_type in self.OPT_CLASS2:
                self.model, self.train_loss, self.test_loss, _, _ = scripts.training.train_pyro_disjoint_param(self.model, train_loader=self.train_loader, test_loader=self.test_loader, verbose=True, num_epochs=max_epochs, convergence_threshold=convergence_threshold, lr=self.optim_args['lr'], eps=self.optim_args['eps'], style="joint", warmup=classifier_warmup)


        # Move model to CPU for evaluation
        # Downstream tasks can move model back to GPU
        self.model = self.model.eval().cpu()
        self.predictive = pyro.infer.Predictive(self.model.generate, num_samples=1)



        # Save the model if desired
        if params_save_path is not None:
            self.model.save(params_save_path)
    
        
        
    def save_model(self,  params_save_path : str):
        """
        Saves the parameters for the currently loaded model.
        """
        self.model.save(params_save_path)


    
    def load_model(self, params_load_path : str ):
        """
        Loads the parameters for the initialized model.
        """
        self.model.load(params_load_path)


    def plot_loss(self, save_loss_path : str = None):
        """
        Plots the training / test losses for the model.
        """
        scripts.visuals.plot_loss(self.train_loss, self.test_loss, save_loss_path=save_loss_path)

    

    def write_embeddings(self): 
        """
        Write latent embeddings to the attached anndata.
        """

        # Add latent generation here per model
        match self.model_type:
            case "SCVI":
                self.anndata.obsm['scvi_latent'] = (self.model.zl_encoder(torch.DoubleTensor(self.dataset[:][0]))[0]).detach().numpy()

            
            case "SCANVI":
                z_latent = self.model.z2l_encoder(torch.DoubleTensor(self.dataset[:][0]))[0]
                z_y = models.basics._broadcast_inputs([z2_latent, self.dataset[:][1]])
                z_y = torch.cat(z2_y, dim=-1)
                u_latent = self.model.z1_encoder(z2_y)[0]

                self.anndata.obsm['scanvi_u_latent'] = u_latent.detach().numpy()
                self.anndata.obsm['scanvi_z_latent'] = z_latent.detach().numpy()

            
            case "Patches":
                rho_latent = self.model.rho_l_encoder(self.dataset[:][0])[0]
                rho_y = models.basics._broadcast_inputs([rho_latent, self.dataset[:][1]])
                rho_y = torch.cat(rho_y, dim=-1)


                w_latent = self.model.w_encoder(rho_y)[0]
                z_latent = self.model.z_encoder(rho_latent)[0]

                self.anndata.obsm['patches_w_latent']  = w_latent.detach().numpy()
                self.anndata.obsm['patches_z_latent']  = z_latent.detach().numpy()

        if self.verbose: print("Written embeddings to object 'anndata.obsm' under workflow.")


    # Evaluate the overall reconstruction error
    def evaluate_reconstruction(self, subset : str = None, n_iter : int = 5):
        printer = []
        
        for metric in self.METRICS_REG.keys():
            if self.verbose : print(f"Calculating {self.METRICS_REG[metric]} ...")
            preds_mean_error, preds_mean_var, pred_profiles, preds = scripts.metrics.get_reproduction_error(self.test_set, self.predictive, metric=metric, n_trials=n_iter, verbose=self.verbose, use_cuda=False, batched=self.batch_correction)

            printer.append(f"{self.METRICS_REG[metric]} : {np.round(preds_mean_error,3)} +- {np.round(preds_mean_var,3)}")


        print("Results\n===================")
        for item in printer: print(item)
            
        
        

        
        



        
class CrossConditionWorkflow(BaseWorkflow):
    """
    Implements the functions for evaluating cross-condition prediction.
    Necessiates the use of a non-linear decoder for high quality transfers.
    """

    # Constructor
    def __init__(self, 
                 anndata : ad.AnnData,
                 verbose : bool = False,
                 random_seed : int = None):

        BaseWorkflow.__init__(self, anndata=anndata, verbose=verbose, config="cross-condition", random_seed=random_seed)



    


    # Evaluate transfers for conditions
    def evaluate_transfer(self, source : str, target : str, n_iter : int = 10):
        """
        Calculates the metrics for quantifying the quality of the transfer, does not require matchings.
        """
        pass
    
    
    