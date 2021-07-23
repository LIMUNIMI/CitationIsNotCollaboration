# FeatGraph
<!--[![GitHub Workflow Status (branch)](https://img.shields.io/github/workflow/status/ChromaticIsobar/featgraph/main/main?event=push)](https://github.com/ChromaticIsobar/featgraph/actions?query=workflow%3Amain)-->
[![Coverage](https://gist.githubusercontent.com/ChromaticIsobar/18a4dd9093b1c271ce6f9d117cc5ba40/raw/featgraph-coverage-badge.svg)](https://github.com/ChromaticIsobar/featgraph/actions?query=workflow%3Amain)
[![Pylint](https://gist.githubusercontent.com/ChromaticIsobar/18a4dd9093b1c271ce6f9d117cc5ba40/raw/featgraph-pylint-badge.svg)](https://github.com/ChromaticIsobar/featgraph/actions?query=workflow%3Amain)
<!--[![PyPI version](https://badge.fury.io/py/featgraph.svg)](https://badge.fury.io/py/featgraph)-->
Musical collaborations graph analysis using WebGraph

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
