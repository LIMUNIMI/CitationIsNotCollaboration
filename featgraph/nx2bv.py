"""Conversion utilities from networkx to BVGraph"""
import networkx as nx
from featgraph import pathutils, logger, conversion
import os
from typing import Optional, Dict, Sequence


def make_asciigraph_txt(
    graph: nx.Graph,
    path: str,
    encoding="utf-8",
    overwrite: bool = False,
):
  """Write the text file of adjacency lists (ASCIIGraph)

  Args:
    graph (Graph): Networkx graph object
    path (str): Destination text file path
    encoding: Encoding for the output file. Default is :data:`"utf-8"`
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  if overwrite or pathutils.notisfile(path):
    with open(path, "w", encoding=encoding) as txt:
      logger.info("Writing ASCIIGraph file: %s", path)
      txt.write(f"{graph.number_of_nodes()}\n")
      for i in range(graph.number_of_nodes()):
        neighbors = sorted(graph[i])
        txt.write(" ".join(map(str, neighbors)) + "\n")


def make_attribute_txt(graph: nx.Graph,
                       path: str,
                       attr: str,
                       missing="",
                       encoding="utf-8",
                       overwrite: bool = False):
  """Write the text file for a node attribute

  Args:
    graph (Graph): Networkx graph object
    path (str): Destination text file path
    attr (str): Attribute key
    missing: Value to print when attribute is missing
    encoding: Encoding for the output file. Default is :data:`"utf-8"`
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  if overwrite or pathutils.notisfile(path):
    with open(path, "w", encoding=encoding) as txt:
      logger.info("Writing Node '%s' file: %s", attr, path)
      for i in range(graph.number_of_nodes()):
        txt.write(f"{graph.nodes[i].get(attr, missing)}\n")


def nx2bv(
    graph: nx.Graph,
    bvgraph_basepath: str,
    attributes: Optional[Dict[str, Sequence[str]]] = None,
    missing="",
    encoding="utf-8",
    overwrite: bool = False,
):
  """Convert a networkx graph to a BVGraph

  Args:
    graph (Graph): Networkx graph object
    bvgraph_basepath (str): Base path for the BVGraph files
    attributes (dict): Attribute names and file suffix for export to text file.
      For each key/value pair, a text file is exported by printing
      for each node the value associated to the key in a file which path is
      derived from :data:`bvgraph_basepath` using the value as suffix
    missing: Value to print when node attribute value is missing
    encoding: Encoding for the output files. Default is :data:`"utf-8"`
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  dirname = os.path.dirname(bvgraph_basepath)
  os.makedirs(dirname, exist_ok=True)
  path = pathutils.derived_paths(bvgraph_basepath)
  make_asciigraph_txt(graph,
                      path("graph-txt"),
                      encoding=encoding,
                      overwrite=overwrite)
  conversion.compress_to_bvgraph(bvgraph_basepath, overwrite=overwrite)
  for k, suffix in attributes.items():
    make_attribute_txt(graph,
                       path(*suffix),
                       k,
                       encoding=encoding,
                       overwrite=overwrite,
                       missing=missing)
