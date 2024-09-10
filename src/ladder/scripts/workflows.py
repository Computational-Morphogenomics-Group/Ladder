import ladder.models as models
import ladder.scripts as scripts
import ladder.data as utils
import anndata as ad
import scanpy as sc
from typing import Literal
import sys, warnings
import pandas as pd
import math
import pyro, torch
import numpy as np


# Force warnings all the time, they are important!
warnings.simplefilter("always", UserWarning)


class BaseWorkflow:
    """
    Implements the base workflow for running any of the pipelines.
    """

    # Static variable for optimizer choice
    OPT_CLASS1 = ["SCVI", "SCANVI"]  # These do not require a complex optimizer loop
    OPT_CLASS2 = ["Patches"]  # These require adversarial optimizer for the latent

    # Static lookup for optimizer defaults
    OPT_DEFAULTS = {
        "lr": 1e-2,
        "eps": 1e-2,
        "betas": (0.90, 0.999),
        "gamma": 1,
        "milestones": [1e10],
    }

    # Static list of keys allowed in optim args
    OPT_LIST = ["optimizer", "optim_args", "gamma", "milestones", "lr", "eps", "betas"]

    # Static list of registered metrics
    # Dict for pretty printing
    METRICS_REG = {
        "rmse": "RMSE",
        "corr": "Profile Correlation",
        "swd": "2-Sliced Wasserstein",
        "chamfer": "Chamfer Discrepancy",
    }

    SEP_METRICS_REG = {
        "knn_error": "kNN Classifier Accuracy",
        "kmeans_nmi": "K-Means NMI",
        "kmeans_ari": "K-Means ARI",
        "calc_asw": "Average Silhouette Width",
    }

    # Constructor
    def __init__(
        self,
        anndata: ad.AnnData,
        config: Literal["cross-condition", "interpretable"] = "cross-condition",
        verbose: bool = False,
        random_seed: int = None,
    ):

        # Set internals
        self.anndata = utils.preprocess_anndata(anndata)
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

        if self.verbose:
            print(f"Initialized workflow to run {config} model.")

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
        return f"<Workflow / Config: {self.config}, Random Seed: {self.random_seed}, Verbose: {self.verbose}, Levels: {self.levels}, Batch Key: {self.batch_key}, Model: {self.model_type}>"

    # Data prep - private call
    def _prep_data(
        self,
        factors: list,
        batch_key: str = None,
        cell_type_label_key: str = None,
        minibatch_size: int = 128,
    ):
        """
        Creates the required data objects to run the model
        """

        if self.model_type is not None:  # Flag cleared, proceed to data setup
            self.label_style = "concat" if self.model_type != "SCANVI" else "one-hot"
            self.factors = factors
            self.len_attrs = [
                len(pd.get_dummies(self.anndata.obs[factor]).columns)
                for factor in factors
            ]

            self.batch_key = batch_key
            self.cell_type_label_key = cell_type_label_key
            self.minibatch_size = minibatch_size

            if self.verbose:
                print(
                    f"\nCondition classes : {self.factors}\nNumber of attributes per class : {self.len_attrs}"
                )

            # Add factorized to anndata column
            self.anndata.obs["factorized"] = [
                " - ".join(row[factor] for factor in self.factors)
                for _, row in self.anndata.obs.iterrows()
            ]
            self.anndata.obs["factorized"] = self.anndata.obs["factorized"].astype(
                "category"
            )

            # Handle batch & create the datasets
            if self.batch_key is not None:
                self.dataset, self.levels, self.converter, self.batch_mapping = (
                    utils.construct_labels(
                        self.anndata.X,
                        self.anndata.obs,
                        self.factors,
                        style=self.label_style,
                        batch_key=self.batch_key,
                    )
                )
                (
                    self.train_set,
                    self.test_set,
                    self.train_loader,
                    self.test_loader,
                    self.l_mean,
                    self.l_scale,
                ) = utils.distrib_dataset(
                    self.dataset, self.levels, batch_size=128, batch_key=self.batch_key
                )

                self.batch_correction = True

            # If no batch
            else:
                self.dataset, self.levels, self.converter = utils.construct_labels(
                    self.anndata.X,
                    self.anndata.obs,
                    self.factors,
                    style=self.label_style,
                )
                (
                    self.train_set,
                    self.test_set,
                    self.train_loader,
                    self.test_loader,
                    self.l_mean,
                    self.l_scale,
                ) = utils.distrib_dataset(
                    self.dataset, self.levels, batch_size=128, batch_key=self.batch_key
                )

                self.batch_correction = False

        else:
            warnings.warn(
                "ERROR: There seems to be no model registered to the workflow. Make sure not to run this function directly if you did so. You must instead run the 'prep_model()' function."
            )

    def _fetch_model_args(self, model_args):
        """
        Used to modify model_args efficiently depending on the model used.
        """
        # For all models
        model_args["reconstruction"] = self.reconstruction
        model_args["batch_correction"] = self.batch_correction
        model_args["scale_factor"] = 1.0 / (
            self.minibatch_size * self.anndata.X.shape[-1]
        )

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
        self.optim_args = {
            k: v for k, v in self.optim_args.items() if k in self.OPT_LIST
        }

    def _register_latent_dims(self):
        """
        Registers latent dimensions to be used downstream
        """
        match self.model_type:
            case self.model_type if self.model_type in self.OPT_CLASS1:
                self.latent_dim = self.model.latent_dim

            case self.model_type if self.model_type in self.OPT_CLASS2:
                self.latent_dim = self.model.latent_dim
                self.w_dim = self.model.w_dim

    def prep_model(
        self,
        factors: list,
        batch_key: str = None,
        cell_type_label_key: str = None,
        minibatch_size: int = 128,
        model_type: Literal["SCVI", "SCANVI", "Patches"] = "Patches",
        model_args: dict = None,
        optim_args: dict = None,
    ):
        """
        Creates the model object to be run.
        """

        # Flush params if needed
        pyro.clear_param_store()

        # Register model type
        self.model_type = model_type
        self.reconstruction = "ZINB" if self.config == "cross-condition" else "ZINB_LD"

        # Prepare the data
        self._prep_data(factors, batch_key, cell_type_label_key, minibatch_size)

        # Grab model constructor
        constructor = getattr(models, self.model_type)

        ## Additional inputs for models

        if model_args is None:
            model_args = {}  ### Get one if not provided

        model_args = self._fetch_model_args(model_args)

        # Construct model
        try:
            self.model = constructor(
                self.anndata.X.shape[-1], self.l_mean, self.l_scale, **model_args
            )

        except Exception as e:
            warnings.warn("\nINFO: model_args ignored, using model defaults...")
            model_args = self._fetch_model_args({})
            self.model = constructor(
                self.anndata.X.shape[-1], self.l_mean, self.l_scale, **model_args
            )

        # Register latents to model
        self._register_latent_dims()

        if self.verbose:
            print(
                f"\nInitialized {self.model_type} model.\nModel arguments: {model_args}"
            )

        # Get optimizer args if not provided
        if optim_args is None:
            optim_args = {}

        # Fill in optimizer gaps
        for key in self.OPT_DEFAULTS.keys():
            if key not in optim_args.keys():
                optim_args[key] = self.OPT_DEFAULTS[key]

        # Grab model optimizer
        match self.model_type:
            case self.model_type if self.model_type in self.OPT_CLASS1:
                self.optim_args = {
                    "optimizer": torch.optim.Adam,
                    "optim_args": {
                        "lr": optim_args["lr"],
                        "eps": optim_args["eps"],
                        "betas": optim_args["betas"],
                    },
                    "gamma": optim_args["gamma"],
                    "milestones": optim_args["milestones"],
                }

            case self.model_type if self.model_type in self.OPT_CLASS2:
                self.optim_args = {
                    "lr": optim_args["lr"],
                    "eps": optim_args["eps"],
                    "betas": optim_args["betas"],
                }

        # Clear whatever is unused
        self._clear_bad_optim_args()

        if self.verbose:
            print(
                f"\nOptimizer args parsed successfully. Final arguments: {self.optim_args}"
            )

    def run_model(
        self,
        max_epochs: int = 1500,
        convergence_threshold: float = 1e-3,
        convergence_window: int = 15,
        classifier_warmup: int = 0,
        params_save_path: str = None,
    ):
        """
        Train the model.
        """

        if dict(pyro.get_param_store()):
            warnings.warn(
                "WARNING: Retraining without resetting parameters is discouraged. Please call prep_model() again if you wish to rerun training."
            )

        if self.verbose:
            print(
                f"Training initialized for a maximum of {max_epochs}, with convergence eps {convergence_threshold}."
            )
        if self.verbose and params_save_path is not None:
            print(f"Model parameters will be saved to path: {params_save_path} ")

        # Match the funtion to run
        match self.model_type:
            case self.model_type if self.model_type in self.OPT_CLASS1:
                self.model, self.train_loss, self.test_loss = scripts.train_pyro(
                    self.model,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader,
                    verbose=True,
                    num_epochs=max_epochs,
                    convergence_threshold=convergence_threshold,
                    convergence_window=convergence_window,
                    optim_args=self.optim_args,
                )

            case self.model_type if self.model_type in self.OPT_CLASS2:
                self.model, self.train_loss, self.test_loss, _, _ = (
                    scripts.train_pyro_disjoint_param(
                        self.model,
                        train_loader=self.train_loader,
                        test_loader=self.test_loader,
                        verbose=True,
                        num_epochs=max_epochs,
                        convergence_threshold=convergence_threshold,
                        convergence_window=convergence_window,
                        lr=self.optim_args["lr"],
                        eps=self.optim_args["eps"],
                        style="joint",
                        warmup=classifier_warmup,
                    )
                )

        # Move model to CPU for evaluation
        # Downstream tasks can move model back to GPU
        self.model = self.model.eval().cpu()
        self.predictive = pyro.infer.Predictive(self.model.generate, num_samples=1)

        # Save the model if desired
        if params_save_path is not None:
            self.model.save(params_save_path)

    def save_model(self, params_save_path: str):
        """
        Saves the parameters for the currently loaded model.
        """
        self.model.save(params_save_path)

    def load_model(self, params_load_path: str):
        """
        Loads the parameters for the initialized model.
        """
        self.model.load(params_load_path)
        self.model = self.model.eval().cpu().double()
        self.predictive = pyro.infer.Predictive(self.model.generate, num_samples=1)

    def plot_loss(self, save_loss_path: str = None):
        """
        Plots the training / test losses for the model.
        """
        scripts._plot_loss(
            self.train_loss, self.test_loss, save_loss_path=save_loss_path
        )

    def write_embeddings(self):
        """
        Write latent embeddings to the attached anndata.
        """

        # Add latent generation here per model
        match self.model_type:
            case "SCVI":
                self.anndata.obsm["scvi_latent"] = (
                    (self.model.zl_encoder(torch.DoubleTensor(self.dataset[:][0]))[0])
                    .detach()
                    .numpy()
                )

            case "SCANVI":
                z_latent = self.model.z2l_encoder(
                    torch.DoubleTensor(self.dataset[:][0])
                )[0]
                z_y = models._broadcast_inputs([z_latent, self.dataset[:][1]])
                z_y = torch.cat(z_y, dim=-1)
                u_latent = self.model.z1_encoder(z_y)[0]

                self.anndata.obsm["scanvi_u_latent"] = u_latent.detach().numpy()
                self.anndata.obsm["scanvi_z_latent"] = z_latent.detach().numpy()

            case "Patches":
                rho_latent = self.model.rho_l_encoder(self.dataset[:][0])[0]
                rho_y = models._broadcast_inputs([rho_latent, self.dataset[:][1]])
                rho_y = torch.cat(rho_y, dim=-1)

                w_latent = self.model.w_encoder(rho_y)[0]
                z_latent = self.model.z_encoder(rho_latent)[0]

                self.anndata.obsm["patches_w_latent"] = w_latent.detach().numpy()
                self.anndata.obsm["patches_z_latent"] = z_latent.detach().numpy()

        if self.verbose:
            print("Written embeddings to object 'anndata.obsm' under workflow.")

    def _subset_by_type(self, cell_type: str):
        """
        Used to subset the test set to a single cell type.
        """

        # Make sure we have that type
        assert cell_type in list(self.anndata.obs[self.cell_type_label_key].astype(str))

        # Do the subset
        if self.verbose:
            print(f"Subsetting test to {cell_type} cells")

        test_subset = self.test_set[
            list(
                np.where(
                    (
                        self.converter.map_to_anndat(self.test_set[:]).obs[
                            self.cell_type_label_key
                        ]
                        == cell_type
                    ).to_numpy()
                )[0]
            )
        ]

        # Cast back into dataset for downstream tasks
        test_subset = torch.utils.data.TensorDataset(*test_subset)

        return test_subset

    # Evaluate the overall reconstruction error
    def evaluate_reconstruction(
        self, subset: str = None, cell_type: str = None, n_iter: int = 5
    ):
        printer = []
        source, target = None, None

        # Grab specific cell type if so
        if cell_type is not None:
            test_set = self._subset_by_type(cell_type)
        else:
            test_set = self.test_set

        # Grab source target if subset
        if subset is not None:
            source, target = torch.DoubleTensor(
                self.levels[subset]
            ), torch.DoubleTensor(self.levels[subset])

        for metric in self.METRICS_REG.keys():
            if self.verbose:
                print(f"Calculating {self.METRICS_REG[metric]} ...")
            preds_mean_error, preds_mean_var, pred_profiles, preds = (
                scripts.metrics.get_reproduction_error(
                    test_set,
                    self.predictive,
                    metric=metric,
                    source=source,
                    target=target,
                    n_trials=n_iter,
                    verbose=self.verbose,
                    use_cuda=False,
                    batched=self.batch_correction,
                )
            )

            printer.append(
                f"{self.METRICS_REG[metric]} : {np.round(preds_mean_error,3)} +- {np.round(preds_mean_var,3)}"
            )

        print("Results\n===================")
        for item in printer:
            print(item)


class InterpretableWorkflow(BaseWorkflow):
    """
    Implements the functions for evaluating the interpretable model.
    Requires a linear decoder by definition.
    """

    # Constructor
    def __init__(
        self, anndata: ad.AnnData, verbose: bool = False, random_seed: int = None
    ):

        BaseWorkflow.__init__(
            self,
            anndata=anndata,
            verbose=verbose,
            config="interpretable",
            random_seed=random_seed,
        )

    def get_conditional_loadings(self):
        """
        Return loadings per condition class or condition string (for SCANVI).
        """

        # TODO: Implement for SCANVI
        assert self.model_type == "Patches"

        # Grab all weights
        mu, logits = self.model.get_weights()

        # Subset to only conditional weights
        mu = mu[self.latent_dim :]

        # Stratify and sum per condition
        cond_latent_ordering = sum(
            [list(self.anndata.obs[factor].cat.categories) for factor in self.factors],
            [],
        )  ## Get latent ordering

        # Set loadings to var
        for k in range(len(cond_latent_ordering)):
            cond_latent = mu[k * self.w_dim : (k + 1) * self.w_dim].sum(dim=0)
            self.anndata.var[f"{cond_latent_ordering[k]}_score_{self.model_type}"] = (
                cond_latent
            )

        if self.verbose:
            print("Written condition specific loadings to 'self.anndata.var'.")

    def get_common_loadings(self):
        """
        Return latent loadings.
        """
        # Grab all weights
        mu, logits = self.model.get_weights()

        # Subset to only common weights
        mu = mu[: self.latent_dim]

        # Set loadings to var
        self.anndata.var[f"common_score_{self.model_type}"] = mu.sum(dim=0)

        if self.verbose:
            print("Written common loadings to 'self.anndata.var'.")


class CrossConditionWorkflow(BaseWorkflow):
    """
    Implements the functions for evaluating cross-condition prediction.
    Necessiates the use of a non-linear decoder for high quality transfers.
    """

    # Constructor
    def __init__(
        self, anndata: ad.AnnData, verbose: bool = False, random_seed: int = None
    ):

        BaseWorkflow.__init__(
            self,
            anndata=anndata,
            verbose=verbose,
            config="cross-condition",
            random_seed=random_seed,
        )

    # Evaluate transfers for conditions
    def evaluate_transfer(
        self, source: str, target: str, cell_type: str = None, n_iter: int = 10
    ):
        """
        Calculates the metrics for quantifying the quality of the transfer, does not require matchings.
        """
        printer = []

        # TODO: Nothing explicit for scVI, implement in future if needed
        assert self.model_type in ("Patches", "SCANVI")

        # Grab specific cell type if so
        if cell_type is not None:
            test_set = self._subset_by_type(cell_type)
        else:
            test_set = self.test_set

        # Check to see the levels actually exist
        # If so, grab
        assert source, target in self.levels
        source_key, target_key = torch.DoubleTensor(
            self.levels[source]
        ), torch.DoubleTensor(self.levels[target])

        if self.verbose:
            print(f"Evaluating mapping...\nSource: {source} --> Target: {target}")

        for metric in self.METRICS_REG.keys():
            if self.verbose:
                print(f"Calculating {self.METRICS_REG[metric]} ...")
            preds_mean_error, preds_mean_var, pred_profiles, preds = (
                scripts.metrics.get_reproduction_error(
                    test_set,
                    self.predictive,
                    metric=metric,
                    source=source_key,
                    target=target_key,
                    n_trials=n_iter,
                    verbose=self.verbose,
                    use_cuda=False,
                    batched=self.batch_correction,
                )
            )

            printer.append(
                f"{self.METRICS_REG[metric]} : {np.round(preds_mean_error,3)} +- {np.round(preds_mean_var,3)}"
            )

        print("Results\n===================")
        for item in printer:
            print(item)

    def evaluate_separability(self, factor: str = None):
        """
        Calculates mixing and separability metrics for latent spaces of the model.
        If factor is supplied, only for that factor. Otherwise for the combination of all factors (as intended).
        """

        # Make sure factor is in factors or not provided
        assert factor is None or factor in self.factors

        # Factor is none means using the factorized column
        if factor is None:
            factor = "factorized"

        # Decide on model
        # Add latent generation here per model
        match self.model_type:
            case "SCVI":
                embed = ["scvi_latent"]

            case "SCANVI":
                embed = ["scanvi_u_latent", "scanvi_z_latent"]

            case "Patches":
                embed = ["patches_w_latent", "patches_z_latent"]

        # Run results for all embeddings
        printer = []

        for emb in embed:
            if self.verbose:
                print(f"Running for embedding: {emb}")

            printer.append(f"\n{emb}\n=========")
            for metric in self.SEP_METRICS_REG.keys():
                func = getattr(scripts, metric)
                printer.append(
                    f"{self.SEP_METRICS_REG[metric]} : {np.round(func(self.anndata, factor, emb),3)}"
                )

        # Print results
        print("Results\n===================")
        for item in printer:
            print(item)
