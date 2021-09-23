"""Test for filter"""
import importlib
import os
import unittest
from tests import testutils
import numpy as np
import featgraph.jwebgraph
from featgraph import conversion


def test_filter_graph(base_path, dest_path, it):
  # load graph
  graph = importlib.import_module("featgraph.jwebgraph.utils").BVGraph(
      base_path)

  # it = filter
  subgraph = graph.transform_map(dest_path, it)
  return subgraph.numNodes()


class TestFilter(
    testutils.TestDataMixin,
    unittest.TestCase,
):
  """Test conversion functions"""
  adjacency_path = "faKedaTa_filter/collaboration_network_edge_list.pickle"
  metadata_path = "faKedaTa_filter/artist_data.pickle"
  base_path = "gRaphZ_filter/testexample-1560"

  def test_whole_filter(self):
    """Test whole filter"""
    tmpdir = ".tmp_f1lt3r_t3st"
    self.adjacency_path = os.path.join(tmpdir, self.adjacency_path)
    self.metadata_path = os.path.join(tmpdir, self.metadata_path)
    self.base_path = os.path.join(tmpdir, self.base_path)
    self.setup_pickles_fn()
    dest_path = self.base_path + "filtered"

    with self.check_files_exist(
        # initial graph
        *testutils.graph_paths(self.base_path),
        self.path("graph-txt"),

        # transformed graph
        *testutils.graph_paths(dest_path),

        # directories and pickles
        *self.pickles_paths()):
      conversion.main(
          self.adjacency_path,
          self.metadata_path,
          self.base_path,
          "--jvm-path",
          testutils.jvm_path,
      )

      # check with files
      np.random.seed(42)
      it = np.greater(np.random.rand(self.nnodes), 0.5)
      n_out = it.sum()
      self.assertEqual(
          n_out,
          featgraph.jwebgraph.jvm_process_run(
              test_filter_graph,
              args=(self.base_path, dest_path, it),
              return_type="i",
              jvm_kwargs=dict(jvm_path=testutils.jvm_path),
          ))
