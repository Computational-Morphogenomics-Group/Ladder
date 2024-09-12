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

    scripts.get_normalized_profile
    scripts.gen_profile_reproduction
    scripts.get_reproduction_error
    scripts.calc_asw
    scripts.kmeans_ari
    scripts.kmeans_nmi
    scripts.knn_error
```

### Training

```{eval-rst}
.. module:: ladder.scripts.training
.. currentmodule:: ladder

.. autosummary::
    :toctree: generated

    scripts.get_device
    scripts.train_pyro
    scripts.train_pyro_disjoint_param
```

### Workflow API

```{eval-rst}
.. module:: ladder.scripts.workflows
.. currentmodule:: ladder

.. autosummary::
    :toctree: generated

    scripts.BaseWorkflow
    scripts.InterpretableWorkflow
    scripts.CrossConditionWorkflow
```
