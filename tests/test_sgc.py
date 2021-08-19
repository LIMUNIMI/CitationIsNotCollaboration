"""Tests for Social Group Centrality model"""
import unittest
from featgraph import sgc


class TestSGC(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.seed = 42
    cls.model = sgc.SGCModel()
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
