"""Tests for Social Group Centrality model"""
import unittest
import networkx as nx
from featgraph import sgc, plots


class TestSGC(unittest.TestCase):
  """Tests Social Group Centrality model"""
  @classmethod
  def setUpClass(cls):
    cls.seed = 42
    cls.model = sgc.SGCModel(n_masses=100)
    cls.graph = cls.model(seed=cls.seed)

  def test_correct_n_nodes(self):
    """Test that the number of nodes is correct"""
    self.assertEqual(
      self.model.n_nodes, self.graph.number_of_nodes()
    )

  def test_correct_n_masses(self):
    """Test that the number of "masses" nodes is correct"""
    self.assertEqual(
      self.model.n_masses,
      sum(1 for _, c in self.graph.nodes(data="class") if c == "masses")
    )

  def test_correct_n_celeb(self):
    """Test that the number of "celebrities" nodes is correct"""
    self.assertEqual(
      self.model.n_celeb,
      sum(1 for _, c in self.graph.nodes(data="class") if c == "celebrities")
    )

  def test_correct_n_leader(self):
    """Test that the number of "community leaders" nodes is correct"""
    self.assertEqual(
      self.model.n_leader,
      sum(
        1 for _, c in self.graph.nodes(data="class") if c == "community leaders"
      )
    )

  def test_plot(self):
    """Test the plot function"""
    plots.draw_sgc_graph(
      self.graph,
    )

  def test_plot_alt(self):
    """Test the plot function with alternative arguments"""
    g = self.graph.copy(as_view=False)

    # add a cross-elite-class edge
    def first_of(k: str) -> int:
      return next(iter(
        i for i, c in g.nodes(data="class")
        if c == k
      ))
    g.add_edge(first_of("celebrities"), first_of("community leaders"))

    plots.draw_sgc_graph(
      g, pos_fn=nx.spectral_layout,
      draw_nodes=True,
    )
