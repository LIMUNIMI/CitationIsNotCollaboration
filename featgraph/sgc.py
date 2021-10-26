"""Implementation of the Directed Social Group Centrality model.
This is the directed version of the Social Group Centrality model described in

*"Popularity and centrality in Spotify networks:
critical transitions in eigenvector centrality"*

https://doi.org/10.1093/comnet/cnaa050"""
import networkx as nx
import pandas as pd
from networkx.generators import random_graphs
from featgraph import nx2bv, pathutils, logger
import functools
import itertools
import importlib
import contextlib
from scipy import stats
import numpy as np
from typing import Optional, Sequence, Callable, Iterable, Tuple, Dict


class SGCModel:
  """Social Group Centrality model for generating random graphs

  Args:
    n_masses (int): Number of nodes in the *masses* class
    n_celeb (int): Number of nodes in the *celebrities* class
    n_leader (int): Number of nodes in the *community leaders* class
    m_masses (int): Number of edges to attach from a new node to existing nodes
      when generating the *masses* subgraph using Barabási–Albert preferential
      attachment
    masses_exp_scale (float): Scale parameter for exponential distribution
      of popularity in the *masses*
    masses_k_pop (float): Popularity threshold for the *masses*. Nodes above
      the threshold will be eligible for connection with *celebrities*. Nodes
      below the threshold will be eligible for connection with *community
      leaders*
    """

  def __init__(
      self,
      n_masses: int = 10000,
      n_celeb: int = 16,
      n_leader: int = 16,
      m_masses: int = 12,
      masses_exp_scale: float = 20.,
      masses_k_pop: float = 50.,
      p_celeb: float = 0.01,
      p_leader: float = 0.1,
      p_celeb_back: float = 1.,
      p_leader_back: float = 1.,
  ):
    self.n_masses = n_masses
    self.n_celeb = n_celeb
    self.n_leader = n_leader
    self.m_masses = m_masses
    self.masses_exp_scale = masses_exp_scale
    self.masses_k_pop = masses_k_pop
    self.p_celeb = p_celeb
    self.p_leader = p_leader
    self.p_celeb_back = p_celeb_back
    self.p_leader_back = p_leader_back

  @property
  def n_nodes(self) -> int:
    """Number of nodes of the output graph"""
    return self.n_masses + self.n_celeb + self.n_leader

  def masses(self, seed: Optional[int] = None) -> nx.DiGraph:
    """Generate the *masses* subgraph using Barabási–Albert preferential
    attachment

    Args:
      seed (int): Seed for random number generator

    Returns:
      DiGraph: The *masses* subgraph"""
    return random_graphs.barabasi_albert_graph(
        n=self.n_masses,
        m=self.m_masses,
        seed=seed,
    ).to_directed(as_view=True)

  def celeb(self):
    """Generate the *celebrities* subgraph as a clique

    Returns:
      DiGraph: The *celebrities* subgraph"""
    return nx.complete_graph(self.n_celeb)

  def leader(self):
    """Generate the *community leaders* subgraph as a clique

    Returns:
      DiGraph: The *community leaders* subgraph"""
    return nx.complete_graph(self.n_leader)

  def popularity(self, seed: Optional[int] = None) -> Sequence[float]:
    """Generate popularity values

    Args:
      seed (int): Seed for random number generator

    Returns:
      array of float: The first values are the popularity values for the
      *masses*, then the values for *celebrities* and *community leaders*"""
    p = np.full(self.n_nodes, 100.)
    p[:self.n_masses] = np.clip(
        stats.expon.rvs(
            scale=self.masses_exp_scale,
            size=self.n_masses,
            random_state=seed,
        ), 0, 100)
    return p

  @staticmethod
  def add_cross_class_edges(
      g: nx.Graph,
      elite_class: str,
      p_any: float,
      p_back_if_any: float = 1.,
      masses_class: str = "masses",
      masses_pop_filter: Optional[Callable[[float], bool]] = None,
      seed: Optional[int] = None,
  ):
    """Add edges across two classes: an *elite* class and the *masses*

    Args:
      g (Graph): Graph to which to add edges
      elite_class (str): Class name of *elite* nodes
      p_any (float): Probability of any edge between a given pair of nodes
      p_back_if_any (float): Probability of an *elite* node to reciprocate
        an existing incoming edge with its symmetric
      masses_class (str): Class name of the *masses*
      masses_pop_filter (callable): Filter function for masses popularity.
        If provided, the *elite* nodes will only be able to connect to the
        *masses* that have a popularity value of :data:`p` for which
        :data:`masses_pop_filter(p)` is :data:`True`
      seed (int): Seed for random number generator"""
    if seed is not None:
      np.random.seed(seed=seed)

    elite_vertices = (i for i, c in g.nodes(data="class") if c == elite_class)

    def mass_vertices() -> Iterable[int]:
      """Return the mass vertices iterable"""
      return (i for i, d in g.nodes(data=True) if d["class"] == masses_class and
              (masses_pop_filter is None or masses_pop_filter(d["popularity"])))

    for ev in elite_vertices:
      for mv in mass_vertices():
        q = np.random.rand()
        if q <= p_any:
          g.add_edge(mv, ev)
          if q <= p_back_if_any:
            g.add_edge(ev, mv)

  def __call__(self, seed: Optional[int] = None) -> nx.DiGraph:
    """Generate a random graph using the Social Group Centrality model

    Args:
      seed (int): Seed for random number generator

    Returns:
      DiGraph: The random graph"""
    masses = self.masses(seed=seed)
    celeb = self.celeb()
    leader = self.leader()

    g = functools.reduce(nx.disjoint_union, (masses, celeb, leader))

    nx.set_node_attributes(
        g,
        name="class",
        values=dict(
            enumerate(
                itertools.chain(
                    itertools.repeat("masses", self.n_masses),
                    itertools.repeat("celebrities", self.n_celeb),
                    itertools.repeat("community leaders", self.n_leader),
                ))),
    )

    nx.set_node_attributes(
        g,
        name="popularity",
        values=dict(enumerate(self.popularity(seed=seed))),
    )

    self.add_cross_class_edges(
        g,
        elite_class="celebrities",
        p_any=self.p_celeb,
        p_back_if_any=self.p_celeb_back,
        masses_pop_filter=lambda p: p > self.masses_k_pop,
        seed=seed,
    )
    self.add_cross_class_edges(
        g,
        elite_class="community leaders",
        p_any=self.p_leader,
        p_back_if_any=self.p_leader_back,
        masses_pop_filter=lambda p: p <= self.masses_k_pop,
        seed=seed,
    )

    g = nx.relabel_nodes(
        g,
        dict(
            enumerate(
                np.random.default_rng(seed=seed).permutation(self.n_nodes))))

    return g


def to_bv(
    graph: nx.Graph,
    bvgraph_basepath: str,
    class_suffix: Sequence[str] = ("type", "txt"),
    popularity_suffix: Sequence[str] = ("popularity", "txt"),
    missing="",
    overwrite: bool = False,
):
  """Convert a SGC networkx graph to a BVGraph

  Args:
    graph (Graph): Networkx graph object
    class_suffix (tuple of str): Suffix for the node class file
    popularity_suffix (tuple of str): Suffix for the node popularity file
    bvgraph_basepath (str): Base path for the BVGraph files
    missing: Value to print when node attribute value is missing
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  nx2bv.nx2bv(graph=graph,
              bvgraph_basepath=bvgraph_basepath,
              missing=missing,
              overwrite=overwrite,
              attributes={
                  "class": class_suffix,
                  "popularity": popularity_suffix,
              })
  return importlib.import_module("featgraph.jwebgraph.utils").BVGraph(
      bvgraph_basepath)


class ThresholdComparison:
  """Class for comparing graphs under threshold

  Args:
    basegraphs: Un-threshold graphs to compare
    thresholds (array): Threshold values
    centralities: Dictionary of centrality names and attribute string
    attribute (str): The attribute to use for thresholding
    attr_fmt: Attribute value formatting function for file saving
    attr_th_fn: Function that takes a threshold values and returns a
      check function"""

  class BaseGraph:
    """Wrapper for a graph to compare

    Args:
      graph: The BVGraph wrapper object
      label (str): The graph's label: If unspecified, the base name is used
      type_key (str): The name of the attribute to read for the node classes
      type_values: Iterable of classes to compare for this graph
      check_fn: Function that is called with a class value as argument and
        returns the checker function for that class"""

    def __init__(self,
                 graph,
                 label: Optional[str] = None,
                 type_key: str = "type_sgc",
                 type_values: Iterable = ("celebrities", "community leaders",
                                          "masses"),
                 check_fn=lambda g: (lambda x: g == x)):
      self.graph = graph
      self.label = self.graph.basename if label is None else label
      self.type_key = type_key
      self.type_values = type_values
      self.check_fn = check_fn

  sgc_graph = BaseGraph

  @classmethod
  def spotify_graph(cls,
                    graph,
                    label: Optional[str] = None,
                    type_key: str = "supergenre",
                    type_values: Iterable = ("classical", "hip-hop", "rock"),
                    check_fn=lambda g: (lambda x: g in x)):
    """Wrapper for a graph to compare, with defaults for the spotify-2018 graph.
    For an explanation of arguments, see :class:`ThresholdComparison.BaseGraph`
    """
    return cls.BaseGraph(graph,
                         label=label,
                         type_key=type_key,
                         type_values=type_values,
                         check_fn=check_fn)

  def __init__(self,
               *basegraphs: Tuple[BaseGraph],
               thresholds: Sequence[float] = tuple(range(0, 81, 5)),
               centralities: Optional[Dict[str, str]] = None,
               attribute: str = "popularity",
               attr_fmt=lambda x: f"{x:02.0f}",
               attr_th_fn=lambda t: (lambda x: x is not None and x > t)):
    self.basegraphs = basegraphs
    self.thresholds = thresholds
    self.centralities = centralities
    if self.centralities is None:
      self.centralities = {
          "Indegree": "indegrees",
          "Harmonic Centrality": "harmonicc",
          "Pagerank": "pagerank",
          "Closeness Centrality": "closenessc",
      }
    self.attribute = attribute
    self.attr_fmt = attr_fmt
    self.attr_th_fn = attr_th_fn

  def subgraph_path(self, base_graph, th) -> str:
    """Get the path of a subgraph for the given base graph and threshold

    Args:
      base_graph (BaseGraph): The base graph to threshold
      th (float): The threshold value

    Returns:
      str: The subgraph path"""
    return base_graph.graph.path(f"{self.attribute}-{self.attr_fmt(th)}")

  def subgraph(self, base_graph, th) -> "BVGraph":
    """Get the subgraph for the given base graph and threshold.
    Note: this method does not compute the subgraph

    Args:
      base_graph (BaseGraph): The base graph to threshold
      th (float): The threshold value

    Returns:
      BVGraph: The subgraph"""
    return importlib.import_module("featgraph.jwebgraph.utils").BVGraph(
        self.subgraph_path(base_graph, th), sep=base_graph.graph.sep)

  def threshold_graphs(self, tqdm: Optional = None, overwrite: bool = False):
    """Compute all thresholded subgraphs

    Args:
      tqdm: function to use for the progress bar
      overwrite (bool): If :data:`False` (default), then skip when the
        output file is found. Otherwise always run"""
    it = itertools.product(self.thresholds, self.basegraphs)
    if tqdm is not None:
      it = tqdm(it, total=len(self.thresholds) * len(self.basegraphs))
    for th, baseg in it:
      subg = self.subgraph(baseg, th)
      logger.info("Thresholding graph %s at %s threshold: %f -> %s",
                  baseg.graph.basename, self.attribute, th, subg.basename)
      if overwrite or pathutils.notisglob(subg.path("*")):
        baseg.graph.transform_map(
            subg.path(),
            map(self.attr_th_fn(th),
                getattr(baseg.graph, self.attribute)()))

  def compute_centralities(self,
                           tqdm: Optional = None,
                           overwrite: bool = False):
    """Compute centralities on all subgraphs

    Args:
      tqdm: function to use for the progress bar
      overwrite (bool): If :data:`False` (default), then skip when the
        output file is found. Otherwise always run"""
    it = itertools.chain(
        itertools.product(self.thresholds, self.basegraphs,
                          (("Transposed graph", "transpose"),)),
        itertools.product(self.thresholds, self.basegraphs,
                          self.centralities.items()),
    )
    if tqdm is not None:
      it = tqdm(it,
                total=len(self.thresholds) * len(self.basegraphs) *
                (len(self.centralities) + 1))
    for th, baseg, (attr_name, attr) in it:
      subg = self.subgraph(baseg, th)
      logger.info("Computing %s for graph %s", attr_name, subg.basename)
      getattr(subg, f"compute_{attr}")(overwrite=overwrite)

  def dataframe(self,
                csv_path: Optional[str] = None,
                overwrite: bool = False,
                tqdm: Optional = None) -> pd.DataFrame:
    """Compute a dataframe with a summary of the subgraph centralities

    Args:
      csv_path (str): Save/load dataframe to/from the csv file at this path
      tqdm: function to use for the progress bar
      overwrite (bool): If :data:`False` (default), then skip when the
        output file is found. Otherwise always run"""
    if csv_path is None or overwrite or pathutils.notisfile(csv_path):
      data = {
          k: [] for k in (
              "graph",
              "threshold",
              "nnodes",
              "narcs",
              "type_key",
              "type_value",
              "centrality",
              "mean",
              "std",
              "quartile-1",
              "median",
              "quartile-3",
          )
      }
      with (contextlib.nullcontext() if tqdm is None else tqdm(
          total=len(self.basegraphs) * len(self.thresholds) *
          len(self.centralities))) as pbar:
        for baseg, th in itertools.product(self.basegraphs, self.thresholds):
          subg = self.subgraph(baseg, th)
          all_types = list(getattr(subg, baseg.type_key)())
          for attr_name, attr in self.centralities.items():
            all_values = np.array(list(getattr(subg, attr)()))
            for tv in baseg.type_values:
              # Add values to dataframe
              data["graph"].append(baseg.label)
              data["threshold"].append(th)
              data["nnodes"].append(subg.numNodes())
              data["narcs"].append(subg.numNodes())
              data["type_key"].append(baseg.type_key)
              data["type_value"].append(tv)
              data["centrality"].append(attr_name)

              values = all_values[list(map(baseg.check_fn(tv), all_types))]
              data["mean"].append(np.mean(values))
              data["std"].append(np.std(values))
              for k, v in zip(("quartile-1", "median", "quartile-3"),
                              np.quantile(values, (0.25, 0.50, 0.75))):
                data[k].append(v)
              del values
            del all_values
            if pbar is not None:
              pbar.update(1)

      df = pd.DataFrame(data=data)
      if csv_path is not None:
        df.to_csv(csv_path)
    else:
      df = pd.read_csv(csv_path, index_col=0)
    df["nnodes_inv"] = 1 / df["nnodes"]
    df["narcs_inv"] = 1 / df["narcs"]
    return df
