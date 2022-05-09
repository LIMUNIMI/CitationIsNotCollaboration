"""Utility wrappers for Java functions.
The JVM should be started before importing this module"""
import json
import functools
import itertools
import jpype
from featgraph import pathutils, metadata, genre_map
from chromatictools import pickle
import shutil
import os
import numpy as np
import pandas as pd
import sys
import featgraph.misc
from featgraph.misc import VectorOrCallable
from typing import Union, List, Sequence, Iterable, Optional, Dict

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
    return f"{type(self).__name__} '{self.basename}' " \
           f"at '{self.base_path}{self.sep}*'"

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

  def compute_degrees(self, overwrite: bool = False, log: bool = True):
    """Compute statistical data of a given graph and save
    indegrees and outdegrees in text format

    Args:
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run
      log (bool): If :data:`True`, then log if file was found"""
    path = self.path("stats")
    if overwrite or pathutils.notisglob(path + "*", log=log):
      webgraph.Stats.main(["--save-degrees", self.base_path, path])

  compute_indegrees = compute_degrees
  compute_outdegrees = compute_degrees

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

  def _copy_metadata(self,
                     other: "BVGraph",
                     suffixes=(
                         ("followers", "txt"),
                         ("genre", "txt"),
                         ("ids", "txt"),
                         ("name", "txt"),
                         ("popularity", "txt"),
                         ("type", "txt"),
                     )):
    """Copy metadata from one graph basename to another

    Args:
      other (BVGraph): The other graph
      suffixes: Metadata file suffixes"""
    for s in suffixes:
      if os.path.isfile(self.path(*s)):
        shutil.copyfile(self.path(*s), other.path(*s))

  def compute_transpose(self, overwrite: bool = False, log: bool = True):
    """Compute the transpose of the graph

    Args:
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run
      log (bool): If :data:`True`, then log if file was found"""
    path = self.path("transpose")
    if overwrite or pathutils.notisglob(path + "*", log=log):
      webgraph.Transform.main(["transposeOffline", self.base_path, path])
      self._copy_metadata(self.transposed())

  def transposed(self) -> "BVGraph":
    """Get the transpose of the graph (must be precomputed)"""
    return type(self)(base_path=self.path("transpose"), sep=self.sep)

  def compute_symmetrized(self, overwrite: bool = False, log: bool = True):
    """Compute the symmetrized graph

    Args:
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run
      log (bool): If :data:`True`, then log if file was found"""
    path = self.path("symmetrized")
    if overwrite or pathutils.notisglob(path + "*", log=log):
      webgraph.Transform.main(["symmetrizeOffline", self.base_path, path])
      self._copy_metadata(self.symmetrized())

  def symmetrized(self) -> "BVGraph":
    """Get the symmetrized graph (must be precomputed)"""
    return type(self)(base_path=self.path("symmetrized"), sep=self.sep)

  def pagerank_path(self, *suffix: str, alpha: float = 0.85):
    r"""Path of PageRank files

    Args:
      suffix: Any number of suffixes
      alpha (float): The :math:`\alpha` value for PageRank

    Returns:
      str: File path"""
    alpha = _pagerank_alpha_preprocess(alpha)
    return self.path(f"pagerank-{100 * alpha:02.0f}", *suffix)

  def compute_pagerank(self,
                       alpha: float = 0.85,
                       overwrite: bool = False,
                       log: bool = True):
    r"""Compute PageRank of a graph given its transpose

    Args:
      alpha (float): PageRank :math:`\alpha` value
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run
      log (bool): If :data:`True`, then log if file was found"""
    alpha = _pagerank_alpha_preprocess(alpha)
    if overwrite or pathutils.notisglob(self.pagerank_path("*", alpha=alpha),
                                        log=log):
      law.rank.PageRankParallelGaussSeidel.main([
          "--alpha",
          f"{alpha:.2f}",
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
                transpose: bool = True,
                log: bool = True,
                overwrite: bool = False):
    r"""Run HyperBall on the graph

    Args:
      command (str): Command flag
      path (str): Output file path
      nbits (int): Number of bits (:math:`\log_2m`) for the
      transpose (bool): Run HyperBall on the transposed graph (default)
      log (bool): If :data:`True`, then log if file was found
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run"""
    graph_paths = (self.path(), self.path("transpose"))
    if transpose:
      graph_paths = reversed(graph_paths)
    if overwrite or pathutils.notisfile(path, log=log):
      webgraph.algo.HyperBall.main([
          "--log2m",
          f"{nbits:.0f}",
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
    self.hyperball(command="-c",
                   path=self.path("closenessc", "ranks"),
                   **kwargs)

  def closenessc(self):
    """Load the Closeness Centrality vector from file

    Returns:
      array of doubles: Array of Closeness Centralities"""
    return load_as_doubles(self.path("closenessc", "ranks"), "Float")

  def compute_linc(self, **kwargs):
    """Compute the Lin Centrality with HyperBall

    Args:
      kwargs: Keyword arguments for :meth:`hyperball`"""
    self.hyperball(command="-L", path=self.path("linc", "ranks"), **kwargs)

  def linc(self):
    """Load the Lin Centrality vector from file

    Returns:
      array of doubles: Array of Lin Centralities"""
    return load_as_doubles(self.path("linc", "ranks"), "Float")

  def compute_reachable(self, **kwargs):
    """Compute the reachable set sizes with HyperBall

    Args:
      kwargs: Keyword arguments for :meth:`hyperball`"""
    self.hyperball(command="-r",
                   path=self.path("reachable", "ranks"),
                   transpose=False,
                   **kwargs)

  def reachable(self):
    """Load the reachable set sizes vector from file

    Returns:
      array of doubles: Array of reachable set sizes"""
    return load_as_doubles(self.path("reachable", "ranks"), "Float")

  def compute_coreachable(self, **kwargs):
    """Compute the coreachable set sizes with HyperBall.
    The transposed graph must be precomputed

    Args:
      kwargs: Keyword arguments for :meth:`hyperball`"""
    self.hyperball(command="-r",
                   path=self.path("coreachable", "ranks"),
                   **kwargs)

  def coreachable(self):
    """Load the reachable set sizes vector from file

    Returns:
      array of doubles: Array of coreachable set sizes"""
    return load_as_doubles(self.path("coreachable", "ranks"), "Float")

  def transform_map(
      self,
      dest_path: str,
      it: Iterable[bool],
      metadata_list: Iterable[str] = ("ids", "type", "name", "popularity",
                                      "genre", "followers")
  ) -> "BVGraph":
    """Filter out the nodes at the indices for which the iterable
    value is :data:`False`

    Args:
      dest_path (str): path where to save the filtered graph
      it (iterable of bool): iterable of boolean values. If :data:`it[i]` is
        :data:`True`, then node :data:`i` is kept, otherwise the node is
        filtered out
      metadata_list (str): list of suffixes of metadata files to map

    Returns:
      a BVGraph which is the filtered graph"""
    n_nodes = int(self.numNodes())
    map_array = jpype.JInt[n_nodes]
    j = 0

    # open metadata files
    src_path = self.base_path + "."
    infnames = list(map(str(src_path + "{}.txt").format, metadata_list))

    # skip files that don't exist
    b = list(map(os.path.exists, infnames))
    infnames = list(itertools.compress(infnames, b))

    outfnames = list(
        itertools.compress(
            map(str(dest_path + ".{}.txt").format, metadata_list), b))

    open_read = functools.partial(open, mode="r", encoding="utf-8")
    open_write = functools.partial(open, mode="w", encoding="utf-8")

    with featgraph.misc.multicontext(map(open_read, infnames)) as infiles:
      with featgraph.misc.multicontext(map(open_write, outfnames)) as outfiles:
        for i, (b, inrows) in enumerate(zip(it, zip(*infiles))):
          if b:
            map_array[i] = j
            for row, outfile in zip(inrows, outfiles):
              outfile.write(row)
            j += 1
          else:
            map_array[i] = -1
    # write
    filtered_graph = webgraph.Transform.map(self.load(), map_array)
    webgraph.BVGraph.store(filtered_graph, dest_path)
    return BVGraph(base_path=dest_path, sep=self.sep)

  def ids(self) -> Iterable[str]:
    """Get the artists ids from the metadata file

    Returns:
      The iterable of ids of each artist as strings"""
    with open(self.path("ids", "txt"), "r", encoding="utf-8") as f:
      for s in f:
        yield s.rstrip()

  def popularity(self, missing_value: Optional = None) -> Iterable[float]:
    """Get the popularity values from the metadata file

      Args:
        missing_value (int): value to assign to an artist in case their
          popularity is unknown

      Returns:
        The iterable of popularity values of each artist
        as floats between 0 and 100 (or :data:`missing_value`)"""
    with open(self.path("popularity", "txt"), "r", encoding="utf-8") as f:
      for r in f:
        s = r.rstrip()
        yield float(s) if s else missing_value

  def genre(self) -> Iterable[List[str]]:
    """Get the genres from the metadata file

      Returns:
        The iterable of the lists of genres of each artist"""
    with open(self.path("genre", "txt"), "r", encoding="utf-8") as f:
      for r in f:
        yield json.loads(r.rstrip("\n"))

  def type_sgc(self) -> Iterable[List[str]]:
    """Get the sgc type from the metadata file

      Returns:
        The iterable of the lists of types of each individual"""
    with open(self.path("type", "txt"), "r", encoding="utf-8") as f:
      for r in f:
        s = r.strip()
        yield str(s) if s else None

  def supergenre(
      self,
      genre_dict: Optional[Dict[str, List[str]]] = None) -> Iterable[List[str]]:
    """Get the supergenres of each artist

    Args:
      genre_dict (dict): The genre map dictionary.
        If :data:`None` (default), the the default map is used.

    Returns:
      The iterable of the lists of supergenres of each artist"""
    return map(
        functools.partial(genre_map.supergenres_from_iterable,
                          genre_map=genre_dict), self.genre())

  def supergenre_dataframe(self,
                           genre_dict: Optional[Dict[str, List[str]]] = None,
                           **kwargs: Dict[str, Iterable]) -> pd.DataFrame:
    """Make a dataframe of the values in the itarables and pair
    it with information about the artist's supergenre. Every artist will have
    one row in the dataframe for each of their supergenres

    Args:
      genre_dict (dict): The genre map dictionary.
        If :data:`None` (default), the the default map is used.
      kwargs (iterables): Named iterables for the dataframe columns

    Returns:
      iterable: The iterable of the lists of supergenre names
    """
    keys = tuple(kwargs.keys())
    values = {k: [] for k in ("aid", "genre", *keys)}
    vals = (itertools.repeat(
        ()),) if len(keys) == 0 else (zip(*[kwargs[k] for k in keys]),)
    for ai, sgi, ti in zip(self.ids(), self.supergenre(genre_dict=genre_dict),
                           *vals):
      for s in sgi:
        values["aid"].append(ai)
        values["genre"].append(s)
        for k, v in zip(keys, ti):
          values[k].append(v)
    return pd.DataFrame(data=values)

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

  def compute_reciprocity(self,
                          overwrite: bool = False,
                          tqdm: Optional = None,
                          log: bool = True):
    """Compute the reciprocity of the arcs of each node
    by counting self-loops and reciprocated arc couples

    Args:
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run
      tqdm: function to use for the progress bar
      log (bool): If :data:`True`, then log if file was found"""
    path = self.path("reciprocity", "dat")
    if overwrite or pathutils.notisfile(path, log=log):
      self_t = BVGraph(self.path("transpose"))

      nit = self.nodeIterator()
      nit_t = self_t.nodeIterator()
      zit = zip(nit, nit_t)
      if tqdm is not None:
        zit = tqdm(zit, total=self.numNodes())

      self_loops = np.zeros(self.numNodes(), dtype=bool)
      couples = np.zeros(self.numNodes(), dtype=int)

      # Check that the position is in the upper triangle of the
      # adjacency matrix, optionally update count of self loops
      def upper_triangle(r, c, update_loops: bool = False):
        if update_loops and c == r:
          nonlocal self_loops
          self_loops[r] = True
          return False
        return c > r

      # Get the iterator of current node successors that correspond
      # to positions in the upper triangle of the adjacency matrix
      def upper_triangle_it(n, it, update_loops: bool = False):
        return filter(
            functools.partial(upper_triangle, n, update_loops=update_loops),
            featgraph.misc.NodeIteratorWrapper(it.successors()))

      for i, _ in zit:
        # assert i == _
        # For each successor in the upper triangle of the matrix,
        # check if it is also in the adjacent matrix
        def in_adjacent(j,
                        adj=tuple(
                            upper_triangle_it(i, nit_t, update_loops=False))):
          return j in adj

        for j in filter(in_adjacent, upper_triangle_it(i,
                                                       nit,
                                                       update_loops=True)):
          couples[i] += 1
          couples[j] += 1
      pickle.save_pickled(dict(
          self_loops=self_loops,
          couples=couples,
      ), path)

  def self_loops(self) -> np.ndarray:
    """Load the self-loops per node from file.
    The file is computed by :meth:`compute_reciprocity`

    Returns:
      array of bool: Self-loops per node"""
    return pickle.read_pickled(self.path("reciprocity", "dat"))["self_loops"]

  def arc_couples(self) -> int:
    """Load the number of reciprocated arcs per node from file.
    The file is computed by :meth:`compute_reciprocity`

    Returns:
      int: Number of reciprocated arc couples per node"""
    return pickle.read_pickled(self.path("reciprocity", "dat"))["couples"]

  def graph_reciprocity(self) -> float:
    """Load the reciprocity stats from file and compute the correlation
    coefficient for the entire graph. The file is computed by
    :meth:`compute_reciprocity`

    Returns:
      float: Reciprocity of the graph"""
    n_entries = self.numNodes() * (self.numNodes() - 1)
    n_valid_arcs = self.numArcs() - self.self_loops().astype(int).sum()

    a_avg = n_valid_arcs / n_entries
    p_cnd = self.arc_couples().sum() / n_valid_arcs

    return (p_cnd - a_avg) / (1 - a_avg)

  def reciprocity(self) -> np.ndarray:
    """Load the reciprocity stats from file and compute the correlation
    coefficient for each node. The file is computed by
    :meth:`compute_reciprocity`. Also, the degrees of each node are needed:
    they are computed by :meth:`compute_degrees`.

    Returns:
      array of float: Reciprocity of each node"""
    n_entries = self.numNodes() - 1
    n_loops = self.self_loops().astype(int)
    n_in = np.array(self.indegrees()).astype(int) - n_loops
    n_out = np.array(self.outdegrees()).astype(int) - n_loops
    del n_loops

    a_avg_in = n_in / n_entries
    a_avg_out = n_out / n_entries
    p_cnd = self.arc_couples() / n_entries

    r = (p_cnd - a_avg_in * a_avg_out) / np.sqrt(a_avg_in * a_avg_out *
                                                 (1 - a_avg_in) *
                                                 (1 - a_avg_out))
    for i, _ in itertools.filterfalse(lambda t: np.isfinite(t[1]),
                                      enumerate(r)):
      r[i] = 0
    return r

  def compute_scc(self,
                  sizes: bool = True,
                  renumber: bool = True,
                  buckets: bool = True,
                  overwrite: bool = False,
                  log: bool = True):
    """Compute strongly connected components

    Args:
      sizes (bool): If :data:`True` (default), then compute component sizes
      sizes (bool): If :data:`True` (default), then renumber components in
        decreasing-size order
      sizes (bool): If :data:`True` (default), then compute buckets (nodes
        belonging to a bucket component, i.e.,
        a terminal nondangling component)
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run
      log (bool): If :data:`True`, then log if file was found"""
    args = []
    paths = []
    if sizes:
      args.append("--sizes")
      paths.append(self.path("scc"))
      paths.append(self.path("sccsizes"))
      paths.append(self.path("node_sccsizes", "dat"))
    if renumber:
      args.append("--renumber")
    if buckets:
      args.append("--buckets")
      paths.append(self.path("buckets"))

    if overwrite or pathutils.notisfile(
        paths, lambda x: all(map(os.path.isfile, x)), log=log):
      webgraph.algo.StronglyConnectedComponents.main([*args, self.base_path])
      node_scc_sizes = np.array(list(
          map(int, map(self.scc_sizes().__getitem__, map(int, self.scc())))),
                                dtype=int)
      pickle.save_pickled(node_scc_sizes, self.path("node_sccsizes", "dat"))

  def scc(self):
    """Load strongly connected components labels vector from file

    Returns:
      array of integers: Array of strongly connected components labels"""
    return load_as_doubles(self.path("scc"), input_type="Integer")

  def scc_sizes(self):
    """Load strongly connected components sizes vector from file

    Returns:
      array of integers: Array of strongly connected components sizes"""
    return load_as_doubles(self.path("sccsizes"), input_type="Integer")

  def node_scc_sizes(self):
    """Load the array of the strongly connected component size of each node

    Returns:
      array of integers: Array of strongly connected component sizes by node"""
    return pickle.read_pickled(self.path("node_sccsizes", "dat"))

  compute_node_scc_sizes = compute_scc

  def compute_wcc(self,
                  sizes: bool = True,
                  renumber: bool = True,
                  buckets: bool = True,
                  overwrite: bool = False,
                  log: bool = True):
    """Compute weakly connected components.
    Requires the symmetrized graph to be computed.

    Args:
      sizes (bool): If :data:`True` (default), then compute component sizes
      sizes (bool): If :data:`True` (default), then renumber components in
        decreasing-size order
      sizes (bool): If :data:`True` (default), then compute buckets (nodes
        belonging to a bucket component, i.e.,
        a terminal nondangling component)
      overwrite (bool): If :data:`False` (default), then skip if the
        output file is found. Otherwise always run
      log (bool): If :data:`True`, then log if file was found"""
    self.symmetrized().compute_scc(sizes=sizes,
                                   renumber=renumber,
                                   buckets=buckets,
                                   overwrite=overwrite,
                                   log=log)

  def wcc(self):
    """Load weakly connected components labels vector from file

    Returns:
      array of integers: Array of weakly connected components labels"""
    return self.symmetrized().scc()

  def wcc_sizes(self):
    """Load weakly connected components sizes vector from file

    Returns:
      array of integers: Array of weakly connected components sizes"""
    return self.symmetrized().scc_sizes()

  def node_wcc_sizes(self):
    """Load the array of the weakly connected component size of each node

    Returns:
      array of integers: Array of weakly connected component sizes by node"""
    return self.symmetrized().node_scc_sizes()

  compute_node_wcc_sizes = compute_wcc
