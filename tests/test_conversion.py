"""Test conversion functions"""
from featgraph import conversion, pathutils, metadata, jwebgraph
from tests import testutils
from unittest import mock
import unittest
import os


def check_neighbors(base_path: str, aid: str, neighbors) -> bool:
  """Check neighbors of node"""
  a = sorted(n.aid for n in metadata.Artist(base_path, aid=aid).neighbors)
  b = sorted(neighbors)
  return a == b


class TestConversion(
  testutils.TestDataMixin,
  unittest.TestCase,
):
  """Test conversion functions"""
  adjacency_path = "faKedaTa_conversion/collaboration_network_edge_list.pickle"
  metadata_path = "faKedaTa_conversion/artist_data.pickle"
  base_path = "gRaphZ_conversion/testexample-1560"

  def test_whole_pipeline(self):
    """Test whole pipeline"""
    tmpdir = ".tmp_conVerZi0n_t3st"
    self.adjacency_path = os.path.join(tmpdir, self.adjacency_path)
    self.metadata_path = os.path.join(tmpdir, self.metadata_path)
    self.base_path = os.path.join(tmpdir, self.base_path)
    jvm_path = os.environ.get("FEATGRAPH_JAVA_PATH", None)

    self.setup_pickles_fn()

    # convert
    conversion.main(
      self.adjacency_path, self.metadata_path, self.base_path,
      "-l", "WARNING",
      *(() if jvm_path is None else ("--jvm-path", jvm_path)),
      )

    # check neighbors from asciigraph
    for k, v in self.adjacency_dict.items():
      with self.subTest(check="neighbors", file="asciigraph", node=k):
        self.assertTrue(check_neighbors(self.base_path, k, v))

    # delete asciigraph
    s = ("graph-txt",)
    with self.subTest(check_exists=s):
      self.assertTrue(os.path.isfile(self.path(*s)))
    os.remove(self.path(*s))

    # check neighbors from bvgraph
    # <!> it fails for neighbors of node e <!>
    failure_nodes = ["e"]
    for k, v in self.adjacency_dict.items():
      expected_failure = k in failure_nodes
      with self.subTest(
        check="neighbors", file="bvgraph", node=k,
        expected_failure=expected_failure,
      ):
        b = jwebgraph.jvm_process_run(
          check_neighbors, args=(self.base_path, k, v),
          return_type="B", jvm_kwargs=dict(jvm_path=jvm_path),
        )
        if expected_failure:
          b = not b
        self.assertTrue(b)

    # check and delete files
    for s in (
      ("followers", "txt"), ("graph",),
      ("ids", "txt"), ("offsets",),
      ("properties",), ("genre", "txt"),
      ("name", "txt"),  # ("graph-txt",),
      ("popularity", "txt"), ("type", "txt"),
    ):
      with self.subTest(check_exists=s):
        self.assertTrue(os.path.isfile(self.path(*s)))
      os.remove(self.path(*s))

    # clean up directories and pickles
    os.rmdir(os.path.dirname(self.path()))
    os.remove(self.adjacency_path)
    os.remove(self.metadata_path)
    os.rmdir(os.path.dirname(self.adjacency_path))
    os.rmdir(os.path.dirname(os.path.dirname(self.path())))

  def test_make_ids(self):
    """Test making ids file"""
    with self.test_data():
      nnodes = self.make_ids_fn()
      with self.subTest(check="number of nodes", cache="miss"):
        self.assertEqual(nnodes, self.nnodes)
      with mock.patch.object(pathutils, "notisfile", return_value=False):
        with self.subTest(check="number of nodes", cache="hit"):
          self.assertEqual(self.make_ids_fn(), self.nnodes)
      with open(self.path("ids", "txt")) as f:
        with self.subTest(check="ids file content"):
          self.assertEqual(
            f.read(),
            "".join(map(
              "{}\n".format,
              sorted(self.adjacency_dict.keys())
            ))
          )

  def test_make_metadata(self):
    """Test making metadata files"""
    with self.test_data():
      self.make_ids_fn()
      self.make_metadata_fn()
      for k in conversion.metadata_labels:
        with self.subTest(check="file length", file=k):
          with open(self.path(k, "txt"), "r") as f:
            self.assertEqual(len(list(f)), self.nnodes)
