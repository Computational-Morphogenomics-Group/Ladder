# Functional API (Low-level)

## Data

### Built-in Data

```{eval-rst}
.. module:: ladder.data.builtin_data
.. currentmodule:: ladder.data

.. autosummary::
    :toctree: generated

    builtin_data.get_data
```

### Tools

```{eval-rst}
.. module:: ladder.data.real_data
.. currentmodule:: ladder.data

.. autosummary::
    :toctree: generated

    real_data.MetadataConverter
    real_data.AnndataConverter
    real_data.construct_labels
    real_data.distrib_dataset
    real_data.preprocess_anndata
```

## Models

```{eval-rst}
.. module:: ladder.models.scvi_variants
.. currentmodule:: ladder.models

.. autosummary::
    :toctree: generated

    scvi_variants.Patches
    scvi_variants.SCVI
    scvi_variants.SCANVI
```

## Scripts

### Metrics

```{eval-rst}
.. module:: ladder.scripts.metrics
.. currentmodule:: ladder.scripts

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
.. currentmodule:: ladder.scripts

.. autosummary::
    :toctree: generated

    training.get_device
    training.train_pyro
    training.train_pyro_disjoint_param
```
