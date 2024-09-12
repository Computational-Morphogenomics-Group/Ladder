# API

## Data

```{eval-rst}
.. module:: ladder.data
.. currentmodule:: ladder

.. autosummary::
    :toctree: generated

    data.MetadataConverter
    data.AnndataConverter
    data.construct_labels
    data.distrib_dataset
    data.preprocess_anndata
```

## Models

```{eval-rst}
.. module:: ladder.models
.. currentmodule:: ladder

.. autosummary::
    :toctree: generated

    models.Patches
    models.SCVI
    models.SCANVI
```

## Scripts

### Metrics

```{eval-rst}
.. module:: ladder.scripts.metrics
.. currentmodule:: ladder

.. autosummary::
    :toctree: generated

    metrics.get_normalized_profile
    metrics.gen_profile_reproduction
    metrics.get_reproduction_error
    metrics.calc_asw
    metrics.kmeans_ari
    metrics.kmeans_nmi
    metrics.knn_error
```

### Training

```{eval-rst}
.. module:: ladder.scripts.training
.. currentmodule:: ladder

.. autosummary::
    :toctree: generated

    training.get_device
    training.train_pyro
    training.train_pyro_disjoint_param
```

### Workflow API

```{eval-rst}
.. module:: ladder.scripts.workflows
.. currentmodule:: ladder

.. autosummary::
    :toctree: generated

    workflows.BaseWorkflow
    workflows.InterpretableWorkflow
    workflows.CrossConditionWorkflow
```
