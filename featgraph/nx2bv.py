"""Conversion utilities from networkx to BVGraph"""
import networkx as nx
from featgraph import pathutils, logger, conversion
import os
from typing import Sequence


def make_asciigraph_txt(
  graph: nx.Graph, path: str,
  overwrite: bool = False,
):
  """Write the text file of adjacency lists (ASCIIGraph)

  Args:
    graph (Graph): Networkx graph object
    path (str): Destination text file path
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  if overwrite or pathutils.notisfile(path):
    with open(path, "w") as txt:
      logger.error("Writing ASCIIGraph file: %s", path)
      txt.write("{}\n".format(graph.number_of_nodes()))
      for i in range(graph.number_of_nodes()):
        neighbors = sorted(graph[i])
        txt.write(" ".join(map(str, neighbors)) + "\n")


def make_class_txt(
  graph: nx.Graph, path: str,
  overwrite: bool = False
):
  """Write the text file of node classes

  Args:
    graph (Graph): Networkx graph object
    path (str): Destination text file path
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  if overwrite or pathutils.notisfile(path):
    with open(path, "w") as txt:
      logger.error("Writing Node class file: %s", path)
      for i in range(graph.number_of_nodes()):
        txt.write("{}\n".format(graph.nodes[i]["class"]))


def nx2bv(
  graph: nx.Graph,
  bvgraph_basepath: str,
  overwrite: bool = False,
  class_suffix: Sequence[str] = ("type", "txt"),
):
  """Convert a networkx graph to a BVGraph

  Args:
    graph (Graph): Networkx graph object
    bvgraph_basepath (str): Base path for the BVGraph files
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file
    class_suffix: Suffix for the node class file"""
  dirname = os.path.dirname(bvgraph_basepath)
  os.makedirs(dirname, exist_ok=True)
  path = pathutils.derived_paths(bvgraph_basepath)
  make_asciigraph_txt(graph, path("graph-txt"), overwrite=overwrite)
  conversion.compress_to_bvgraph(bvgraph_basepath, overwrite=overwrite)
  make_class_txt(graph, path(*class_suffix), overwrite=overwrite)
