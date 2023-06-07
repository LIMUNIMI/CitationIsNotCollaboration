# Citation is not Collaboration <!-- omit in toc -->
*Music-Genre Dependence of Graph-Related Metrics in a Music Credits Network*
<!--[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/LIMUNIMI/CitationIsNotCollaboration/main/main?event=push)](https://github.com/LIMUNIMI/CitationIsNotCollaboration/actions?query=workflow%3Amain)-->
[![Coverage](https://gist.githubusercontent.com/ChromaticIsobar/18a4dd9093b1c271ce6f9d117cc5ba40/raw/featgraph-coverage-badge.svg)](https://github.com/LIMUNIMI/CitationIsNotCollaboration/actions?query=workflow%3Amain)
[![Pylint](https://gist.githubusercontent.com/ChromaticIsobar/18a4dd9093b1c271ce6f9d117cc5ba40/raw/featgraph-pylint-badge.svg)](https://github.com/LIMUNIMI/CitationIsNotCollaboration/actions?query=workflow%3Amain)
<!--[![PyPI version](https://badge.fury.io/py/featgraph.svg)](https://badge.fury.io/py/featgraph)-->

## Table of Contents <!-- omit in toc -->
- [Publications](#publications)
- [Setup](#setup)
  - [Create environment](#create-environment)
  - [Add extra dependencies](#add-extra-dependencies)
  - [Add FeatGraph](#add-featgraph)
- [Notebooks](#notebooks)
- [CLI](#cli)
  - [Data conversion](#data-conversion)

## Publications
We published some of the results in the article *``Citation is not Collaboration: Music-Genre Dependence of Graph-Related Metrics in a Music Credits Network''* at the 20th Sound and Music Computing Conference.
```bibtex
@inproceedings{clerici2023citation,
  author       = {Clerici, Giulia and Tiraboschi, Marco},
  title        = {{Citation is not Collaboration: Music-Genre Dependence of Graph-Related Metrics in a Music Credits Network}},
  booktitle    = {Proceedings of the 20th Sound and Music Computing Conference},
  year         = {2023},
  series       = {SMC},
  address      = {Stockholm, Sweden},
  month        = {6},
  organization = {Sound and Music Computing Network},
}
```

## Setup
Some dependencies require [`conda`](https://conda.io).  
The following instructions assume that you are working from the root directory of the repository

### Create environment
There seems to be some issues with conda when trying to install too many packages.  
So we spit the installation requirements in chunks.

Create the environment
```bash
conda create -p ./venv python=3.8
```
Activate the enviroment
```bash
conda activate ./venv
```
Install packages in chunks
```bash
conda install -c chromaticisobar --file requirements-0of6.txt -y && \
conda install -c conda-forge     --file requirements-1of6.txt -y && \
conda install -c conda-forge     --file requirements-2of6.txt -y && \
conda install -c conda-forge     --file requirements-3of6.txt -y && \
conda install -c conda-forge     --file requirements-4of6.txt -y && \
conda install -c conda-forge     --file requirements-5of6.txt -y
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
