"""Test conversion functions"""
from pyfakefs import fake_filesystem_unittest
from chromatictools import pickle
from featgraph import conversion, pathutils
import functools
import unittest
from unittest import mock
import numpy as np
import os
from typing import Optional


def random_string(n: int, m: Optional[int] = None, g: float = 1.0) -> str:
  """Generate a random string

  Args:
    n (int): Minimum length of the string
    m (int): Maximum length of the string. Defaults to :data:`None`
      (same as minimum)
    g (float): Exponent for string length distribution"""
  if m is None:
    m = n
  return "".join(np.random.choice(
    list(map(chr, range(ord("a"), ord("z") + 1))),
    int(m + (np.random.rand()**g) * (n - m))
  ))


class TestConversion(unittest.TestCase):
  """Test conversion functions"""
  def setUp(self):
    self.adjacency_path = "faKedaTa/collaboration_network_edge_list.pickle"
    self.metadata_path = "faKedaTa/artist_data.pickle"

    self.adjacency_dict = {
      "a": ["b", "e"],
      "b": ["c"],
      "c": ["b", "d"],
      "d": ["c", "f"],
      "e": ["a", "b"],
      "f": ["e"],
    }
    self.nnodes = len(self.adjacency_dict)
    np.random.seed(42)
    metadata = [
      # popularity
      (np.random.rand(self.nnodes) * 100).astype(int),
      # genre
      [
        [random_string(3, 6, 0.5) for _ in range(int(np.random.rand() * 4))]
        for _ in range(self.nnodes)
      ],
      # name
      [random_string(4, 12, 2).capitalize() for _ in range(self.nnodes)],
      # type
      np.full(self.nnodes, "artist"),
      # followers
      (np.random.rand(self.nnodes)**2 * 1000).astype(int),
    ]
    self.metadata = [
      dict(zip(self.adjacency_dict.keys(), v))
      for v in metadata
    ]

    self.base_path = "gRaphZ/testexample-1560"
    self.path = pathutils.derived_paths(self.base_path)

  def test_whole_pipeline(self):
    """Test whole pipeline"""
    jvm_path = os.environ.get("FEATGRAPH_JAVA_PATH", None)
    tmpdir = ".tmp_conVerZi0n_t3st"
    adjacency_path = os.path.join(tmpdir, self.adjacency_path)
    pickle.save_pickled(self.adjacency_dict, adjacency_path)
    metadata_path = os.path.join(tmpdir, self.metadata_path)
    pickle.save_pickled(self.metadata, metadata_path)
    base_path = os.path.join(tmpdir, self.base_path)
    conversion.main(
      adjacency_path, metadata_path, base_path,
      "-l", "WARNING",
      *(() if jvm_path is None else ("--jvm-path", jvm_path)),
      )
    path = pathutils.derived_paths(base_path)
    for s in (
      ("followers", "txt"), ("graph",),
      ("ids", "txt"), ("offsets",),
      ("properties",), ("genre", "txt"),
      ("graph-txt",),  ("name", "txt"),
      ("popularity", "txt"), ("type", "txt"),
    ):
      with self.subTest(check_exists=s):
        self.assertTrue(os.path.isfile(path(*s)))
      os.remove(path(*s))
    # clean up
    os.rmdir(os.path.dirname(path()))
    os.remove(adjacency_path)
    os.remove(metadata_path)
    os.rmdir(os.path.dirname(adjacency_path))
    os.rmdir(os.path.dirname(os.path.dirname(path())))

  def test_make_ids(self):
    """Test making ids file"""
    with fake_filesystem_unittest.Patcher():
      pickle.save_pickled(self.adjacency_dict, self.adjacency_path)
      os.makedirs(os.path.dirname(self.base_path), exist_ok=True)
      make_ids = functools.partial(
        conversion.make_ids_txt,
        self.path("ids", "txt"),
        self.adjacency_path,
        lambda x: x,
      )
      nnodes = make_ids()
      with self.subTest(check="number of nodes", cache="miss"):
        self.assertEqual(nnodes, self.nnodes)
      with mock.patch.object(pathutils, "notisfile", return_value=False):
        with self.subTest(check="number of nodes", cache="hit"):
          self.assertEqual(make_ids(), self.nnodes)
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
    with fake_filesystem_unittest.Patcher():
      pickle.save_pickled(self.adjacency_dict, self.adjacency_path)
      pickle.save_pickled(self.metadata, self.metadata_path)
      os.makedirs(os.path.dirname(self.base_path), exist_ok=True)
      conversion.make_ids_txt(
        self.path("ids", "txt"),
        self.adjacency_path,
      )
      conversion.make_metadata_txt(
        self.path(),
        self.metadata_path,
        self.path("ids", "txt"),
        lambda x: x,
      )
      for k in conversion.metadata_labels:
        with self.subTest(check="file length", file=k):
          with open(self.path(k, "txt"), "r") as f:
            self.assertEqual(len(list(f)), self.nnodes)
