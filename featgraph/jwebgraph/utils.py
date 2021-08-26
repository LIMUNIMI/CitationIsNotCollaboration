"""Utility wrappers for Java functions.
The JVM should be started before importing this module"""
import json

import jpype

from featgraph import pathutils, metadata
import os
import numpy as np
import sys
import featgraph.misc
from featgraph.misc import VectorOrCallable
from typing import Union, List, Sequence, Iterable
# imports from java
try:
  from it.unimi.dsi import law, webgraph
  import java.lang
except ModuleNotFoundError as e:
  if os.path.basename(sys.argv[0]) != "sphinx-build":
    raise ModuleNotFoundError("Java modules not found") from e


def jaccard(a: Sequence[metadata.Artist],
            b: Sequence[metadata.Artist]) -> float:
  """Jaccard index between sets of artists

  Args:
    a: First set of artists
    b: Second set of artists

  Returns:
    float: Jaccard index"""
  return featgraph.misc.jaccard(
      [x.index for x in a],
      [x.index for x in b],
  )


def load_as_doubles(
    path: str,
    input_type: Union[type, str] = "String",
    reverse: bool = False,
):
  """Loads a vector of doubles, either in binary or textual form.

  Args:
    path (str): The file path
    input_type (str or type): The input type (:class:`Double`, :class:`Float`,
      :class:`Integer`, :class:`Long` or :class:`String` to denote a text file).
      Default is :class:`String`.
      Either the type object can be passed or its name as a string
    reverse (bool): Whether to reverse the ranking induced by the score vector
      by loading opposite values

  Returns:
    array of double: The data read from the file"""
  if isinstance(input_type, str):
    input_type = getattr(java.lang, input_type)
  return law.stat.KendallTau.loadAsDoubles(path, input_type, reverse)


def kendall_tau(x: VectorOrCallable, y: VectorOrCallable):
  r"""Computes Kendall's :math:`\tau` between two score vectors.

  Note that this method must be called with some care.
  More precisely, the two arguments should be built on-the-fly in
  the method call, and not stored in variables, as the first argument
  array will be null'd during the execution of this method to free
  some memory: if the array is referenced elsewhere the garbage
  collector will not be able to collect it

  Args:
    x: The first vector or a function that returns the first vector
    y: The second vector or a function that returns the second vector

  Returns:
    double: Kendall's :math:`\tau`"""
  return law.stat.KendallTau.INSTANCE.compute(
      x() if callable(x) else x,
      y() if callable(y) else y,
  )


def _pagerank_alpha_preprocess(alpha: float, a: float = 0.01, b: float = 0.99):
  r"""Preprocess PageRank :math:`\alpha` value

  Args:
    alpha (float): :math:`\alpha` value
    a (float): Minimum accepted value
    b (float): Maximum accepted value

  Returns:
    float: Clipped :math:`\alpha` value"""
  return max(min(alpha, b), a)


class BVGraph:
  """BVGraph wrapper class

  Args:
    base_path (str): Base path for graph files
    sep (str): Separator for graph file suffixes"""

  def __init__(self, base_path: str, sep: str = "."):
    self.base_path = base_path
    self.sep = sep
    self._loaded = None

  def path(self, *suffix: str) -> str:
    """Derived paths for graph files

    Args:
      suffix: Any number of suffixes"""
    return pathutils.derived_paths(self.base_path, self.sep)(*suffix)

  @property
  def basename(self) -> str:
    """Base name of the graph"""
    return os.path.basename(self.base_path)

  def __str__(self) -> str:
    """Pretty-print graph name and path pattern"""
    return "{} '{}' at '{}{}*'".format(
        type(self).__name__, self.basename, self.base_path, self.sep)

  @property
  def loaded(self) -> bool:
    """Whether the java BVGraph has been loaded or not"""
    return self._loaded is not None

  def load(self) -> "webgraph.BVGraph":
    """Load (if not already loaded) and return the java BVGraph

    Returns:
      webgraph.BVGraph: The java BVGraph"""
    if not self.loaded:
      self._loaded = webgraph.BVGraph.load(self.base_path)
    return self._loaded

  def __getattr__(self, item: str):
    """Delegate attribute misses to the java BVGraph"""
    return getattr(self.load(), item)

  def reconstruct_offsets(self, overwrite: bool = False):
    """Generate offsets for the source graph

    Args:
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run"""
    path = self.path("offsets")
    if overwrite or pathutils.notisfile(path):
      webgraph.BVGraph.main(["-O", self.base_path])

  def compute_degrees(self, overwrite: bool = False):
    """Compute statistical data of a given graph and save
    indegrees and outdegrees in text format

    Args:
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run"""
    path = self.path("stats")
    if overwrite or pathutils.notisglob(path + "*"):
      webgraph.Stats.main(["--save-degrees", self.base_path, path])

  def indegrees(self):
    """Load indegrees vector from file

    Returns:
      array of doubles: Array of indegrees"""
    return load_as_doubles(self.path("stats", "indegrees"))

  def outdegrees(self):
    """Load outdegrees vector from file

    Returns:
      array of doubles: Array of outdegrees"""
    return load_as_doubles(self.path("stats", "outdegrees"))

  def compute_transpose(self, overwrite: bool = False):
    """Compute the transpose of the graph

    Args:
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run"""
    path = self.path("transpose")
    if overwrite or pathutils.notisglob(path + "*"):
      webgraph.Transform.main(["transposeOffline", self.base_path, path])

  def pagerank_path(self, *suffix: str, alpha: float = 0.85):
    r"""Path of PageRank files

    Args:
      suffix: Any number of suffixes
      alpha (float): The :math:`\alpha` value for PageRank

    Returns:
      str: File path"""
    alpha = _pagerank_alpha_preprocess(alpha)
    return self.path("pagerank-{:02.0f}".format(100 * alpha), *suffix)

  def compute_pagerank(self, alpha: float = 0.85, overwrite: bool = False):
    r"""Compute PageRank of a graph given its transpose

    Args:
      alpha (float): PageRank :math:`\alpha` value
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run"""
    alpha = _pagerank_alpha_preprocess(alpha)
    if overwrite or pathutils.notisglob(self.pagerank_path("*", alpha=alpha)):
      law.rank.PageRankParallelGaussSeidel.main([
          "--alpha",
          "{:.2f}".format(alpha),
          self.path("transpose"),
          self.pagerank_path(alpha=alpha),
      ])

  def pagerank(self, alpha: float = 0.85):
    r"""Load PageRank values vector from file

    Args:
      alpha (float): PageRank :math:`\alpha` value

    Returns:
      array of doubles: Array of PageRank values"""
    return load_as_doubles(self.pagerank_path("ranks", alpha=alpha), "Double")

  def hyperball(self,
                command: str,
                path: str,
                nbits: int = 8,
                transpose: bool = True):
    r"""Run HyperBall on the graph

    Args:
      command (str): Command flag
      path (str): Output file path
      nbits (int): Number of bits (:math:`\log_2m`) for the
      transpose (bool): Run HyperBall on the transposed graph (default)"""
    graph_paths = (self.path(), self.path("transpose"))
    if transpose:
      graph_paths = reversed(graph_paths)
    if pathutils.notisfile(path):
      webgraph.algo.HyperBall.main([
          "--log2m",
          "{:.0f}".format(nbits),
          "--offline",
          "--external",
          command,
          path,
          *graph_paths,
      ])

  def compute_neighborhood(self, **kwargs):
    r"""Compute the neighborhood function with HyperBall

    Args:
      kwargs: Keyword arguments for :meth:`hyperball`"""
    self.hyperball(command="-n", path=self.path("nf", "txt"), **kwargs)

  def neighborhood(self):
    """Load neighborhood fuction vector from file

    Returns:
      array of doubles: Array of cumulative frequencies"""
    return load_as_doubles(self.path("nf", "txt"))

  def distances(self):
    """Compute the distance distribution vector from the
    neighborhood fuction file

    Returns:
      array of doubles: Array of absolute frequencies"""
    n = self.neighborhood()
    a = np.zeros(len(n) + 1)
    a[1:] = n
    return np.diff(a)

  def compute_harmonicc(self, **kwargs):
    """Compute the Harmonic Centrality with HyperBall

    Args:
      kwargs: Keyword arguments for :meth:`hyperball`"""
    self.hyperball(command="-h", path=self.path("hc", "ranks"), **kwargs)

  def harmonicc(self):
    """Load the Harmonic Centrality vector from file

    Returns:
      array of doubles: Array of Harmonic Centralities"""
    return load_as_doubles(self.path("hc", "ranks"), "Float")

  def compute_closenessc(self, **kwargs):
    """Compute the Closeness Centrality with HyperBall

    Args:
      kwargs: Keyword arguments for :meth:`hyperball`"""
    self.hyperball(command="-c", path=self.path("closenessc", "ranks"), **kwargs)

  def closenessc(self):
    """Load the Closeness Centrality vector from file

    Returns:
      array of doubles: Array of Closeness Centralities"""
    return load_as_doubles(self.path("closenessc", "ranks"), "Float")

  # do not need, delete after transform_map is ready!
  def map_nodes(self, filtered_nodes):
    """Create the map_array, which is a list to be passed to transform_map to filter the graph.
       If map[i] == -1, the node is removed. Else, assign to the nodes to keep an incremental value starting from zero.

        Args:
          n_nodes (int): the total number of nodes in the original graph
          filtered_nodes (int list): a list containing the indices of the nodes of the graph to keep
        Returns:
          map (jpype.JInt[]): an array containing the a value for each node, whether it is removed or not"""
    n_nodes = self.numNodes()
    map_array = jpype.JInt[n_nodes]
    c = 0
    for i in range(n_nodes):
      if i not in filtered_nodes:
        map_array[i] = -1
      else:
        map_array[i] = c
        c += 1
    return map_array

  # TODO
  # es. it = filter(lambda p: p > 10, self.popularity())
  # def transform_map(self, dest_path: str, it: Iterable[bool]) -> "BVGraph":
  #   a = init java array
  #   j = 0
  #   for i, f in enumerate(it):
  #     if f:
  #       a[i] = j
  #       j += 1
  #       print metadata to files
  #     else:
  #       a[i] = -1
  #       skip metadata rows
  #   transform graph
  #   store graph
  #   return BVGraph(base_path=dest_path, sep=self.sep)

  def transform_map(self, dest_path: str, it: list, overwrite: bool = False) -> "BVGraph":
    """Transform a graph according to the mapping in map_array. If map[i] == -1, the node is removed.

        Args:
          dest_path (str): path where to save the filtered graph
          it (Iterable[bool]): iterable containing a Boolean value indicating if the node is kept or removed
          overwrite (bool): bool to indicate if the function overwrites the existing files or not
        Returns:
          a BVGraph which is the filtered graph
          """
    metadata_list = ["ids", "type", "name", "popularity", "genre", "followers"]
    new_path = dest_path + '.'
    d_r_files = {}
    d_w_files = {}
    for key in metadata_list:
      src_path = self.base_path + '.' + key
      new_path = dest_path + '.' + key
      d_r_files[src_path] = open(src_path + '.txt', 'r')
      d_w_files[new_path] = open(new_path + '.txt', 'w')
      src_path = src_path.replace(key, '')
      new_path = new_path.replace(key, '')
    n_nodes = self.numNodes()
    map_array = jpype.JInt[n_nodes]
    j = 0
    for i in n_nodes:
      if i in it:
        map_array[i] = j
        j += 1
    #       print metadata to files - line by line
        for key in metadata_list:
          src_path = src_path + '.' + key
          new_path = dest_path + '.' + key
          cont = d_r_files[src_path].readlines()[i]
          d_w_files[new_path].write(cont)
          cont.clear()
      else:
        map_array[i] = -1
    for key in metadata_list:
      src_path = self.base_path + '.' + key
      new_path = dest_path + '.' + key
      d_r_files[src_path].close()
      d_w_files[new_path].close()
    webgraph.Transform.map(self.load(), map_array)
    path = dest_path  # self.path("map-" + key)
    if overwrite or pathutils.notisglob(path + "*"):
      webgraph.BVGraph.store(self, path) # or self.load() ?
    return BVGraph(base_path=dest_path, sep=self.sep)

  def popularity(self, missing_value: int):
    """Function that retrieve the popularity inside the popularity.txt file of the graph

        Args:
          missing_value (int): value to assign to an artist in case her popularity is unknown
        Returns:
          the string (str) containing the popularities of the artists of the graph"""
    missing_value = str(missing_value)
    with open(self.path("popularity", "txt"), "r") as f:
      return [float(r.rstrip("\n") or missing_value) for r in f]

  def genre(self):
    """Function that retrieve the genre inside the genre json file of the graph

        Returns:
          the json containing the genres of the artists of the graph"""
    with open(self.path("genre", "txt"), "r") as f:
      return [json.loads(r.rstrip("\n")) for r in f]

  def popularity_filter(self, threshold):
    """Filter the graph nodes based on thei popularity, according to the specified threshold

        Args:
          threshold (int): the threshold to filter the artists based on their popularity values
        Returns:
          filtered_nodes (int list): the list containing the indices of the filtered nodes"""
    pop_values = self.popularity(missing_value=-20)
    filtered_nodes = list(
      filter(lambda i: (pop_values[i] > threshold), range(len(pop_values))))
    return filtered_nodes

  def genre_filter(self, threshold, key):
    """Filter the graph nodes based on their genre, according to the specified threshold

        Args:
          threshold (int): the threshold to filter the artists based on their genre values
          key (str): a str that indicates if we want to select the nodes considering AND or OR to evaluate the genre
              values of an artist
        Returns:
          filtered_nodes (int list): the list containing the indices of the filtered nodes"""
    genre_values = self.genre()
    if key == 'and':
      filtered_nodes = list(
        filter(lambda i: (all(x in genre_values[i] for x in threshold)),
               range(len(genre_values))))
    elif key == 'or':
      filtered_nodes = list(
        filter(lambda i: (any(x in genre_values[i] for x in threshold)),
               range(len(genre_values))))
    return filtered_nodes

  def centrality(self, type_c):
    """Filter the graph nodes based on their genre, according to the specified threshold

      Args:
        type_c (str): a str indicating which kind of centrality to consider
      Returns:
        c_values (int list): a list containing the (specified) centrality values for each artist"""
    if type_c == "pagerank":
      c_values = self.pagerank()
    elif type_c == "hc":
      c_values = self.harmonicc()
    elif type_c == "closenessc":
      c_values = self.closenessc()
    else:
      print("Unknown centrality")
    return c_values

  def centrality_filter(self, type_c, threshold):
    """Filter the graph nodes based on their centrality, according to the specified threshold

        Args:
          type_c (str): a str indicating the type of centrality to consider
          threshold (int): the threshold to filter the artists based on their centrality values
        Returns:
          filtered_nodes (int list): the list containing the indices of the filtered nodes"""
    c_values = self.centrality(type_c)
    filtered_nodes = list(
      filter(lambda i: (c_values[i] > threshold), range(len(c_values))))
    return filtered_nodes

  def update_txt_metadata(self, src_path, key, type_filter, threshold, lines_to_keep):
    """Function for the creation of the metadata file for the filtered graph:

    Args:
        src_path: source path of the metadata file of the original graph
        key: the type of metadata you want to retrieve
        type_filter: the type of filter you used to filter the graph
        threshold: the threshold for type_filter you used to filter the graph
        lines_to_keep: the indices of the nodes you filtered and for which you want to retrieve the metadata
    """
    new_path = src_path + '.mapped-' + type_filter + '-' + str(threshold) + '.' + key
    print("Creating file: %s" % (new_path))
    src_path = src_path + '.' + key
    fn = open(src_path + '.txt', 'r')
    fn1 = open(new_path + '.txt', 'w')

    cont = fn.readlines()
    for i in range(0, len(cont)):
      if i in lines_to_keep:
        fn1.write(cont[i])
      else:
        pass

    fn.close()
    fn1.close()

  def artist(self, **kwargs) -> metadata.Artist:
    """Get an artist from the dataset

    Args:
      kwargs: Keyword arguments for :class:`featgraph.metadata.Artist`

    Returns:
      Artist: the Artist object wrapper"""
    return metadata.Artist(self.base_path, **kwargs)

  def best(self,
           n: int,
           f: VectorOrCallable,
           reverse: bool = True) -> List[metadata.Artist]:
    """Compute the best-scoring nodes for a function

    Args:
      n (int): The number of nodes to return
      f: A vector of scores or a callable that returns a vector of scores
      reverse (bool): If :data:`True` (default) compute maxima, otherwise
        compute minima

    Returns:
      list of :class:`featgraph.metadata.Artist`: The best artists for the score
"""
    if callable(f):
      f = f()
    arg = np.argsort(f)
    if reverse:
      arg = reversed(arg[-n:])
    else:
      arg = arg[:n]
    return [self.artist(index=i) for i in arg]


def store_subgraph(graph, subgraph_path, overwrite: bool = True):
    """Store the properties file of the subgraph

        Args:
          overwrite (bool): If :data:`False` (default), then skip if the
            output file is found. Otherwise always run"""
    path = subgraph_path #self.path("map-" + key)
    if overwrite or pathutils.notisglob(path + "*"):
      webgraph.BVGraph.store(graph, path)

# TODO
# filter_metric(graph.harmonicc(), ...)
# def filter_metric(metric, it):
#  for m, b in zip(metric, it):
#    if b:
#      yield m

