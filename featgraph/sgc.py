"""Implementation of the Directed Social Group Centrality model.
This is the directed version of the Social Group Centrality model described in

*"Popularity and centrality in Spotify networks:
critical transitions in eigenvector centrality"*

https://doi.org/10.1093/comnet/cnaa050"""
import networkx as nx
from networkx.generators import random_graphs
import functools
import itertools
from scipy import stats
import numpy as np
from typing import Optional, Sequence, Callable, Iterable


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
      ), 0, 100
    )
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

    elite_vertices = (
      i for i, c in g.nodes(data="class")
      if c == elite_class
    )

    def mass_vertices() -> Iterable[int]:
      """Return the mass vertices iterable"""
      return (
        i for i, d in g.nodes(data=True)
        if d["class"] == masses_class and (
          masses_pop_filter is None or masses_pop_filter(d["popularity"])
        )
      )

    for ev in elite_vertices:
      for mv in mass_vertices():
        q = np.random.rand()
        if q <= p_any:
          g.add_edge(mv, ev)
          if q <= p_back_if_any:
            g.add_edge(ev, mv)

  def __call__(self, seed: Optional[int] = None):
    """Generate a random graph using the Social Group Centrality model

    Args:
      seed (int): Seed for random number generator

    Returns:
      DiGraph: The random graph"""
    masses = self.masses(seed=seed)
    celeb = self.celeb()
    leader = self.leader()

    g = functools.reduce(
      nx.disjoint_union,
      (masses, celeb, leader)
    )

    nx.set_node_attributes(
      g, name="class",
      values=dict(enumerate(itertools.chain(
        itertools.repeat("masses", self.n_masses),
        itertools.repeat("celebrities", self.n_celeb),
        itertools.repeat("community leaders", self.n_leader),
      ))),
    )

    nx.set_node_attributes(
      g, name="popularity",
      values=dict(enumerate(self.popularity(seed=seed))),
    )

    self.add_cross_class_edges(
      g, elite_class="celebrities",
      p_any=self.p_celeb, p_back_if_any=self.p_celeb_back,
      masses_pop_filter=lambda p: p > self.masses_k_pop,
      seed=seed,
    )
    self.add_cross_class_edges(
      g, elite_class="community leaders",
      p_any=self.p_leader, p_back_if_any=self.p_leader_back,
      masses_pop_filter=lambda p: p <= self.masses_k_pop,
      seed=seed,
    )

    g = nx.relabel_nodes(
      g, dict(enumerate(
        np.random.default_rng(seed=seed).permutation(self.n_nodes)
      ))
    )

    return g
