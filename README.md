# FeatGraph
<!--[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/ChromaticIsobar/featgraph/main/main?event=push)](https://github.com/ChromaticIsobar/featgraph/actions?query=workflow%3Amain)-->
[![Coverage](https://gist.githubusercontent.com/ChromaticIsobar/18a4dd9093b1c271ce6f9d117cc5ba40/raw/featgraph-coverage-badge.svg)](https://github.com/ChromaticIsobar/featgraph/actions?query=workflow%3Amain)
[![Pylint](https://gist.githubusercontent.com/ChromaticIsobar/18a4dd9093b1c271ce6f9d117cc5ba40/raw/featgraph-pylint-badge.svg)](https://github.com/ChromaticIsobar/featgraph/actions?query=workflow%3Amain)
<!--[![PyPI version](https://badge.fury.io/py/featgraph.svg)](https://badge.fury.io/py/featgraph)-->

Musical collaborations graph analysis using WebGraph

## Setup
Some dependencies require [`conda`](https://conda.io).
You should
 - [create a virtual environment](#create-environment)
 - [add extra dependencies](#add-extra-dependencies) (optional, for notebooks or for development)
 - [add the `featgraph` package](#add-featgraph)

The following instructions assume that you are working from the root directory of the repository

### Create environment
To setup the working environment create a new environment using the yaml specifications

```
conda env create --prefix ./venv -f environment.yml
```

This will create a `conda` environment with runtime dependencies in the subfolder `venv` of your working directory.

To activate the environment, run

```
conda activate ./venv
```

### Add extra dependencies
You can add extra development dependencies using the requirements files.

```
conda install --file <FILE> -c conda-forge
```

 - `test-requirements.txt`: Tests dependencies
 - `docs-requirements.txt`: Docs generation dependencies
 - `style-requirements.txt`: Style check dependencies
 - `notebooks-requirements.txt`: Notebooks dependencies

To install all dependencies, run

```
conda install -c conda-forge --file test-requirements.txt --file docs-requirements.txt --file style-requirements.txt --file notebooks-requirements.txt
```

### Add FeatGraph
To add the `featgraph` package, run

```
conda develop .
```

## Notebooks
Interactive notebooks are in the [notebooks](notebooks) folder

## CLI
Some functionalities are available to the command-line

### Data conversion
You can convert the original pickled dataset in BVGraph binary files and
plain-text metadata with
```
python -m featgraph.conversion
```
```
usage: python -m featgraph.conversion [-h] [--jvm-path PATH] [-l LEVEL] [--tqdm] adjacency_path metadata_path dest_path

Convert original pickled dataset into text and BVGraph files

positional arguments:
  adjacency_path        The path of the adjacency lists pickle file
  metadata_path         The path of the metadata pickle file
  dest_path             The destination base path for the BVGraph and text files

optional arguments:
  -h, --help            show this help message and exit
  --jvm-path PATH       The Java virtual machine full path
  -l LEVEL, --log-level LEVEL
                        The logging level. Default is 'INFO'
  --tqdm                Use tqdm progress bar (you should install tqdm for this)
```
