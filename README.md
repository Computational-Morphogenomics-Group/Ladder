# Patches
Logo goes here

<a href="https://github.com/Computational-Morphogenomics-Group/Ladder/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/Computational-Morphogenomics-Group/Ladder/actions/workflows/test.yaml/badge.svg?branch=main"></a>
<a href="https://ladder.readthedocs.io"><img alt="Docs" src="https://img.shields.io/readthedocs/Ladder"></a>
<a href="https://github.com/Computational-Morphogenomics-Group/ladder/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a> <!-- Courtesy of black docs for now -->
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>


Patches takes as input a collection of observations or gene expression profiles obtained from multiple conditions referred to as groups -- for example, age, treatment or injury stage. Each group is characterized by a unique set of categories or attributes -- for example, `young' or `old' represent attributes of the group `age'. Given an anndata object with metadata for the observations contained in separate columns of anndata.obs, Patches can:

- Learn conditional and condition agnostic latent representations for observations.
- 'Transfer' observations across combinations of conditions.
- Provide attribute specific scores, associating gene expression with effects of attributes.

embed pictures here

## Table of Contents

1. [Installation](#installation)
2. [Repository structure](#repository-structure)
3. [Getting started](#getting-started)
5. [Release notes](#release-notes)
6. [Issues and contact](#issues-and-contact)
7. [License and citation](#license-and-citation)

## Installation
Build the environment with:
```
conda env create -f env.yml
```

Install everything with:

```
pip install git+https://github.com/Computational-Morphogenomics-Group/Ladder.git@patches
```

## Repo Structure
Alongside Patches, this repository provides the low and high-level API's (called Ladder, still in development at time of Patches release) for running similar models with data from multiple conditions built on anndata.
- All code is contained under the src directory. (add modules and stuff here)
- Tests are contained under the tests directory.
- All documentation can be generated and is serviced through the docs directory.

### Workflow API (for basic users)
The workflow API provides a very easy interface to apply models considered in the study ([scVI](https://scvi-tools.org/), [scANVI](https://scvi-tools.org/), Patches; all built on the Pyro framework) to multicondition datasets easily.
- This is recommended for users that want to directly apply the model, without worrying much for technical details.
- Consult the workflow docs for hyperparameters you can define and how to define them.

```python
from ladder.data import get_data
from ladder.scripts import CrossConditionWorkflow, InterpretableWorkflow # Workflow APIs, read docs on which one you'd like for your case!

# Initialize workflow object
workflow = InterpretableWorkflow(anndata, verbose=True, random_seed=42)

# Define the condition classes & batch key to prepare the data
factors = ["time", "age", "broad_type"]
workflow.prep_model(factors, batch_key="sample", model_type='Patches', model_args={'ld_normalize' : True}) # Define model and optimizer hyperparams

# Train the model
workflow.run_model(max_epochs=2000, convergence_threshold=1e-5, convergence_window=2000) # Define convergence hyperparams
```

### Developer API (for users who wish to build on the code provided)
The low-level developer API explicitly performs all actions condensed into workflows. This allows for:
- Implementing and running additional models using the Pyro framework.
- Data preparation functions to provide ease for running the models and downstream visualizations.
- Evaluation of generative accuracy alongside mixing/separation of conditional effects using defined metrics.

This API is designed to be specifically used by those who are interested in specialized applications or those who would like to extend the codebase. Please use the issue tracker for any questions / encountered problems.

## Getting started

Please check out the [documentation][https://ladder.readthedocs.io] for all instructions and tutorials.

## Release notes

See the [changelog][changelog].

## Issues and contact

For questions and help requests, please reach out to [PUT CONTACT HERE][contact-address].
If you encounter any bugs, please report it through the [issue tracker][issue-tracker]. In general, providing a minimally reproducible example is the best way to facilitate easy debugging of issues.

## Citation
This project is licensed under the terms of the MIT License. If the codebase has been of any use to you, please cite:

> TBD

[contact-address]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/Computational-Morphogenomics-Group/Ladder/issues
[changelog]: https://ladder.readthedocs.io/latest/changelog.html
[link-api]: https://ladder.readthedocs.io/latest/api.html
[link-pypi]: https://pypi.org/project/ladder
