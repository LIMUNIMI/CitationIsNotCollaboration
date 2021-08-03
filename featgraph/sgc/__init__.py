"""Implementation of the Directed Social Group Centrality model.
This is the directed version of the Social Group Centrality model described in

*"Popularity and centrality in Spotify networks:
critical transitions in eigenvector centrality"*

https://doi.org/10.1093/comnet/cnaa050"""
import networkx as nx
from networkx.generators import random_graphs
import functools
from typing import Optional


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

  def __call__(self, seed=None):
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

    return g
