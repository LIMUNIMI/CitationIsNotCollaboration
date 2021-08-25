"""Tests for plots"""
import unittest
import importlib
from tests import testutils
from featgraph import jwebgraph
import numpy as np


def test_scatter(*args, **kwargs):
  """Test scatterplot"""
  importlib.import_module("featgraph.plots").scatter(*args, **kwargs)


class TestPlots(unittest.TestCase):
  """Tests for plots"""

  def setUp(self):
    self.ndots = 5
    np.random.seed(42)
    self.x = np.random.randn(self.ndots)
    self.y = np.random.randn(self.ndots)

  def test_scatter(self):
    """Test scatterplot"""
    jwebgraph.jvm_process_run(test_scatter,
                              jvm_kwargs=dict(jvm_path=testutils.jvm_path),
                              args=(
                                  self.x,
                                  self.y,
                              ),
                              kwargs=dict(
                                  label="random scatter",
                                  xlabel="N_0",
                                  ylabel="N_1",
                              ))

  def test_scatter_nokt(self):
    """Test scatterplot with no Kendall tau"""
    jwebgraph.jvm_process_run(test_scatter,
                              jvm_kwargs=dict(jvm_path=testutils.jvm_path),
                              args=(
                                  self.x,
                                  self.y,
                              ),
                              kwargs=dict(kendall_tau=False))
