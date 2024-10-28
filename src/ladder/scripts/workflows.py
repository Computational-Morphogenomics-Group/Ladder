"""The workflows module houses the workflow API.

The workflow API is intended as a high-level API for users to
generally apply the models included in `ladder` to their datasets.
It includes standard applications of the models defined within.
"""

import math
import warnings
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
import pyro
import torch

import ladder.data as utils
import ladder.models as models
import ladder.scripts as scripts

# Force warnings all the time, they are important!
warnings.simplefilter("always", UserWarning)


class BaseWorkflow:
    """Base class for all workflows.

    Offers a high-level API that does not require running blocks of
    code in quick succession, as the process for each dataset is more or
    less similar. Must not be instantiated and used directly. All parameters
    given to specific functions throughout the workflow can later be accessed with
    the same named attribute.

    Parameters
    ----------
    anndata : :class:`~anndata.AnnData`
        The dataset object to be used throughout the analyses.

    config : :class:`Literal["cross-condition", "interpretable"]`, default: "cross-condition"
        Defines the workflow to be used. Affects model structure.

    verbose : :class:`bool`, default: False
        If `True`, prints progress messages for various methods within the module.

    random_seed : :class:`int`, optional
        If given, seeds the internal modules with the value.

    Attributes
    ----------
    anndata : :class:`~anndata.AnnData`
        The attached :class:`~torch.utils.data.Dataset` object.

    batch_key : :class:`str`, optional
        Optional batch key in :attr:`~anndata.AnnData.obs` for correction.

    batch_mapping : :class:`dict`
        Mapping of batch literals to encodings, only appears if batch key is provided in workflow.

    cell_type_label_key : :class:`str`, optional
        Optional cell type labels in :attr:`~anndata.AnnData.obs`, required if cell-type specific evaluation is desired.

    config : :class:`Literal["cross-condition", "interpretable"]`, default: "cross-condition"
        The config string provided during construction.

    converter : :class:`~ladder.data.real_data.AnndataConverter`
        Low-level converter class for the attached :class:`~torch.utils.data.Dataset`. See `:func:`~ladder.data.real_data.distrib_dataset` for details.

    dataset : :class:`~torch.utils.data.Dataset`
        Low-level :class:`~torch.utils.data.Dataset` object passed to the model. See `:func:`~ladder.data.real_data.distrib_dataset` for details.

    factors : :class:`list`
        List of factors to register to the model.

    verbose : :class:`bool`, optional
        If `True`, prints progress messages for various methods within the module.

    random_seed : :class:`int`, optional
        If given, seeds the internal modules with the value.

    label_style : :class:`str`
        Defines the conditional encoding style to use depending on the model.

    latent_dim : :class:`int`
        Size of the latent dimension for the model. Common latent for Patches.

    len_attrs : :class:`list`
        Specifies the number of attributes per condition class.

    levels : :class:`dict`
        Mapping of condition literals to encodings.

    l_mean : :class:`float` or array_like
        If :attr:`batch_key` is provided in workflow, the empirical library size log-mean for each batch (1-D Array-like of :class:`float`). A single value otherwise.

    l_scale : :class:`float` or array_like
        If :attr:`batch_key` is provided in workflow, then the empirical library size log-variance for each batch (1-D Array-like of :class:`float`). A single value otherwise.

    minibatch_size : :class:`int`
        Size of the minibatch to be provided during training.

    model : :class:`~torch.nn.Module`
        The model object attached to the workflow.

    model_type : :class:`str`
        Specifies the model attached to the current workflow.

    optim_args : :class:`dict`
        Optimizer arguments passed to low-level trainer. See :mod:`~ladder.scripts.training` for details.

    predictive : :class:`~pyro.infer.predictive.Predictive`
        Low-level generator to be used for tasks after training.

    reconstruction : :class:`str`
        Defines the decoder to be used.

    train_loss : :class:`~numpy.ndarray`
        :class:`~numpy.ndarray` of losses recorded on the training set during training.

    train_set : :class:`~torch.utils.data.Dataset`
        Low-level training :class:`~torch.utils.data.Dataset` passed to the model. See `:func:`~ladder.data.real_data.distrib_dataset` for details.

    test_loss : :class:`~numpy.ndarray`
        :class:`~numpy.ndarray` of losses recorded on the test set during training.

    test_set : torch.utils.data.Dataset
        Low-level test :class:`~torch.utils.data.Dataset` passed to the model. See `:func:`~ladder.data.real_data.distrib_dataset` for details.

    w_dim : :class:`int`, optional
        Size of conditional latents, only defined for Patches.

    Methods
    -------
    prep_model(factors, batch_key=None, cell_type_label_key=None, minibatch_size=128, model_type="Patches", model_args=None, optim_args=None)
        Prepares the model to be run.

    run_model(max_epochs=1500, convergence_threshold=1e-3, convergence_window=30, classifier_warmup=0, params_save_path=None)
        Runs the model on the attached data object.

    save_model(params_save_path)
        Saves the attached model.

    load_model(params_load_path)
        Loads parameters for the attached model. Needs :meth:`prep_model` to be run first.

    plot_loss()
        Simple plotter for loss functions.

    write_embeddings()
        Places the calculated cell embeddings from the trained model under the corresponding :attr:`~anndata.AnnData.obsm` field.

    evaluate_reconstruction(subset=None, cell_type=None, n_iter=5)
        Evaluates the quality of reconstructions with generative metrics.

    evaluate_separability(factor=None)
        Evaluates the separability of the latent encodings with respect to conditional effects.

    """

    # Static variable for optimizer choice
    OPT_CLASS1 = ["SCVI", "SCANVI"]  # These do not require a complex optimizer loop
    OPT_CLASS2 = ["Patches"]  # These require adversarial optimizer for the latent

    # Static lookup for optimizer defaults
    OPT_DEFAULTS = {
        "lr": 1e-3,
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

    # Printer
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

    # Repr
    def __repr__(self):
        return f"<Workflow / Config: {self.config}, Random Seed: {self.random_seed}, Verbose: {self.verbose}, Levels: {self.levels}, Batch Key: {self.batch_key}, Model: {self.model_type}>"

    # Data preparation, implicitly called by prep_model
    # For specific functions see the related module
    def _prep_data(
        self,
        factors: list,
        batch_key: str = None,
        cell_type_label_key: str = None,
        minibatch_size: int = 128,
    ):
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
                    self.dataset,
                    self.levels,
                    batch_size=self.minibatch_size,
                    batch_key=self.batch_key,
                    drop_last=True,
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
                    self.dataset,
                    self.levels,
                    batch_size=self.minibatch_size,
                    batch_key=self.batch_key,
                    drop_last=True,
                )

                self.batch_correction = False

        else:
            warnings.warn(
                "ERROR: There seems to be no model registered to the workflow. Make sure not to run this function directly if you did so. You must instead run the 'prep_model()' function.",
                stacklevel=2,
            )

    # Set up model args
    def _fetch_model_args(self, model_args):
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

    # Delete unused optimizer args
    def _clear_bad_optim_args(self):
        self.optim_args = {
            k: v for k, v in self.optim_args.items() if k in self.OPT_LIST
        }

    # Register latent dims to attributes
    def _register_latent_dims(self):
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
        """Prepares the model to be run.

        The choice of model implicitly decides the kind of condition encodings
        to use, so there is no need to have a separate data preparation.

        Parameters
        ----------
        factors : :class:`list`
            Factors from :attr:`~anndata.AnnData.obs` to register to the model.

        batch_key : :class:`str`, optional
            Defines the workflow to be used. Affects model structure. Can later be accessed with same named attribute.

        cell_type_label_key : :class:`str`, optional
            Optional cell type labels in :attr:`~anndata.AnnData.obs`, required if cell-type specific evaluation is desired.

        minibatch_size : :class:`int`, default: 128
            Size of the minibatch to be provided during training.

        model_type : :class:`Literal["SCVI", "SCANVI", "Patches"]`, default: "Patches"
            Specifies the model attached to the current workflow.

        model_args : :class:`dict`
            Model arguments passed to low-level model constructor. See :mod:`~ladder.models` for details.

        optim_args : :class:`dict`
            Optimizer arguments passed to low-level trainer. See :mod:`~ladder.scripts.training` for details.
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

        except (TypeError, ValueError) as e:
            print(f"Exception encountered while passing model args: {e}")
            warnings.warn(
                "\nINFO: model_args ignored, using model defaults...", stacklevel=2
            )
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

        # Fill optim args attr
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

        # Clear whatever is unused
        self._clear_bad_optim_args()

        if self.verbose:
            print(
                f"\nOptimizer args parsed successfully. Final arguments: {self.optim_args}"
            )

    def run_model(
        self,
        max_epochs: int = 1500,
        convergence_threshold: float = 1e-4,
        convergence_window: int = 100,
        classifier_warmup: int = 0,
        classifier_aggression: int = 0,
        params_save_path: str = None,
    ):
        """Runs the model on the attached data object.

        Parameters
        ----------
        max_epochs : :class:`int`, default: 1500
            Maximum number of epochs to run.

        convergence_threshold : :class:`float`, default: 1e-3
            Minimum improvement required to continue training.

        convergence_window : :class:`int`, default: 30
            Number of epochs to wait until a new minimum is attained.

        classifier_warmup : :class:`int`, default: 0
            Number of epochs to run the classifier before running the entire model.

        classifier_aggression : :class:`int`, default: 0
            Number of epochs the classifier takes independently between jointly trained epochs. Used for Patches.

        params_save_path : :class:`str`, optional
            If provided, saves the model to the specified path.

        """
        if dict(pyro.get_param_store()):
            warnings.warn(
                "WARNING: Retraining without resetting parameters is discouraged. Please call prep_model() again if you wish to rerun training.",
                stacklevel=2,
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
                        warmup=classifier_warmup,
                        classifier_aggression=classifier_aggression,
                        optim_args=self.optim_args,
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
        """Saves the attached model.

        Parameters
        ----------
        params_save_path : :class:`str`
            Path to save model parameters. Expects only the name without extensions.
        """
        self.model.save(params_save_path)

    def load_model(self, params_load_path: str):
        """Loads parameters for the attached model. Needs :meth:`prep_model` to be run first.

        Parameters
        ----------
        params_load_path : :class:`str`
            Path to find model parameters. Expects only the shared prefix, and not the trailing "_torch.pth" or "_pyro.pth".
        """
        self.model.load(params_load_path)
        self.model = self.model.eval().cpu().double()
        self.predictive = pyro.infer.Predictive(self.model.generate, num_samples=1)

    def plot_loss(self, save_loss_path: str = None):
        """Simple plotter for loss functions.

        Parameters
        ----------
        save_loss_path : :class:`str`, optional
            If provided, saves the figure to the specified location. Requires the full name with extensions (eg. fig.png).
        """
        scripts._plot_loss(
            self.train_loss, self.test_loss, save_loss_path=save_loss_path
        )

    def write_embeddings(self):
        """Places the calculated cell embeddings from the trained model under the corresponding :attr:`~anndata.AnnData.obsm` field.

        Each model has a separate name for their respective latent, so that more than a
        single workflow running on the same object instance does not overwrite info.
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

                if self.reconstruction not in ["ZINB_LD", "Normal_LD"]:
                    self.anndata.obsm["patches_rho_latent"] = (
                        rho_latent.detach().numpy()
                    )

        if self.verbose:
            print("Written embeddings to object 'anndata.obsm' under workflow.")

    # Subset the test set to a single type
    def _subset_by_type(self, cell_type: str):
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
        """Evaluates the quality of reconstructions with generative metrics.

        Parameters
        ----------
        subset : :class:`str`, optional
            Key from :attr:`~BaseWorkflow.levels` to subset cells for a specific condition before evaluating reconstruction.

        cell_type : :class:`str`, optional
            Requires :attr:`~BaseWorkflow.cell_type_label_key` to be defined as attribute. Subset cells to a single type before evaluating reconstruction.

        n_iter : :class:`int`, default: 5
            Number of times to repeat the generative process.
        """
        return_dict, printer = {}, []
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

            return_dict[self.METRICS_REG[metric]] = [
                np.round(preds_mean_error, 3),
                np.round(preds_mean_var, 3),
            ]

        print("Results\n===================")
        for item in printer:
            print(item)

        return return_dict

    def evaluate_separability(self, factor: str = None):
        """Evaluates the separability of latent embeddings for conditions.

        Parameters
        ----------
        factor : :class:`str`, optional
            Item listed in :attr:`BaseWorkflow.factors`. If not provided, the metrics will be evaluated on the combinations of factors.
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


class InterpretableWorkflow(BaseWorkflow):
    """Interpretable workflow for training with a linear decoder.

    Inherits :class:`BaseWorkflow` and adds functionalities desired from
    running the interpretable models with linear decoders.

    Parameters
    ----------
    anndata : :class:`~anndata.AnnData`
        The dataset object to be used throughout the analyses.

    verbose : :class:`bool`, default: False
        If `True`, prints progress messages for various methods within the module.

    random_seed : :class:`int`, optional
        If given, seeds the internal modules with the value.

    Methods
    -------
    get_conditional_loadings()
        Writes attribute specific gene loadings to :attr:`~anndata.AnnData.var`.

    get_common_loadings()
        Writes non-conditional gene loadings to :attr:`~anndata.AnnData.var`.
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
        """Writes attribute specific gene loadings to :attr:`~anndata.AnnData.var`.

        Only to be used with Patches, as the other models do not offer
        an attribute-specific way to learn coefficients.
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
        """Writes non-conditional gene loadings to :attr:`~anndata.AnnData.var`.

        Can be used with all models.
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
    """Cross-condition workflow for training with a non-linear decoder.

    Inherits :class:`BaseWorkflow` and adds functionalities desired from
    running a cross-conditional model for more precise reconstructions and transfers.

    Parameters
    ----------
    anndata : :class:`~anndata.AnnData`
        The dataset object to be used throughout the analyses.

    verbose : :class:`bool`, default: False
        If `True`, prints progress messages for various methods within the module.

    random_seed : :class:`int`, optional
        If given, seeds the internal modules with the value.

    Methods
    -------
    evaluate_transfer(source, target, cell_type=None, n_iter=10)
        Evaluates the quality of transfers with generative metrics.
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
        """Evaluates the quality of transfers with generative metrics.

        Parameters
        ----------
        source : :class:`str`
            Key from :attr:`BaseWorkflow.levels` to decide source condition.

        target : :class:`str`
            Key from :attr:`BaseWorkflow.levels` to decide target condition.

        cell_type : :class:`str`, optional
            Requires :attr:`BaseWorkflow.cell_type_label_key` to be defined as attribute. Subset cells to a single type before evaluating transfer.

        n_iter : :class:`int`, default: 10
            Number of times to repeat the generative process.
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
