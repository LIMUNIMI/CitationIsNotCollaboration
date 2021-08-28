"""Test for filter"""
import random
import unittest
import  featgraph.jwebgraph.utils
from pyfakefs import fake_filesystem_unittest


class TestFilter(unittest.TestCase):
  """Tests for the filter functions in utils"""

  def test_transform_map(self, basename):
    graph = featgraph.jwebgraph.utils.BVGraph(basename)
    type_filt = "popularity"
    random.seed(42)
    r = random.randint(0, int(graph.numNodes()))
    thresh_filt = graph.artist(index=r).popularity
    dest_path = basename + ".test-unittest-subgraph." + type_filt + "-" + str(thresh_filt)
    map_array = list(map(lambda p: p > thresh_filt, graph.popularity(-20)))
    subgraph = graph.transform_map(dest_path, map_array)
    n = int(graph.numNodes())
    m = int(subgraph.numNodes())
    self.assertEqual(sum(map_array), m)


  def test_write_metadata(self, basename):
    # load a graph
    graph = featgraph.jwebgraph.utils.BVGraph(basename)
    # create an iterable [bool] to store the results of the filtering of some property
    missing_value = -20
    random.seed(42)
    r = random.randint(0, int(graph.numNodes()))
    thresh_p = graph.artist(index=r).popularity
    map_array = list(map(lambda p: p > thresh_p, graph.popularity(missing_value)))
    dest_path = basename + ".test-unittest-subgraph.popularity"
    subgraph = graph.transform_map(dest_path, map_array)
    graph.write_metadata_files(self, map_array, dest_path)
    j = 0
    for m, b in zip(graph.popularity(), map_array):
      if b:
        self.assertEqual(m, subgraph.artist(index=j).popularity)
        j += 1

    random.seed(42)
    r = random.randint(0, int(graph.numNodes()))
    thresh_g = graph.artist(index=r).genre
    map_array = list(map(lambda g: g in thresh_g, graph.genre()))
    dest_path = basename + ".test-unittest-subgraph.genre"
    subgraph = graph.transform_map(dest_path, map_array)
    graph.write_metadata_files(self, map_array, dest_path)
    j = 0
    for m, b in zip(graph.genre(), map_array):
      if b:
        self.assertEqual(m, subgraph.artist(index=j).genre)
        j += 1

    random.seed(42)
    r = random.randint(0, int(graph.numNodes()))
    type_centrality = "hc"
    thresh_c = graph.centrality(type_centrality)[r]
    map_array = list(map(lambda c: c in thresh_c, graph.centrality(type_centrality)))
    dest_path = basename + ".test-unittest-subgraph." + type_centrality
    subgraph = graph.transform_map(dest_path, map_array)
    graph.write_metadata_files(self, map_array, dest_path)
    c_sub = subgraph.centrality(type_centrality)
    j = 0
    for m, b in zip(graph.centrality(type_centrality), map_array):
      if b:
        self.assertEqual(m, c_sub[j])
        j += 1
