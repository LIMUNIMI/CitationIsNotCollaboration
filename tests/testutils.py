"""Utilities for tests"""
import numpy as np
from pyfakefs import fake_filesystem_unittest
from chromatictools import pickle
from featgraph import conversion, pathutils
import contextlib
import os
from typing import Optional


jvm_path = os.environ.get("FEATGRAPH_JAVA_PATH", None)


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


def random_metadata(n: int):
  """Generate random metadata

  Args:
    n (int): Number of nodes"""
  return [
    # popularity
    (np.random.rand(n) * 100).astype(int),
    # genre
    [
      [
        random_string(3, 6, 0.5)
        for _ in range(int(np.random.rand() * 4))
      ] for _ in range(n)
    ],
    # name
    [
      random_string(4, 12, 2).capitalize()
      for _ in range(n)
    ],
    # type
    np.full(n, "artist"),
    # followers
    (np.random.rand(n) ** 2 * 1000).astype(int),
  ]


def test_data(seed: int = 42):
  """Get test data and metadata

  Args:
    seed (int): RNG seed"""
  adjacency_dict = {
    "a": ["b", "e"],
    "b": ["c"],
    "c": ["b", "d"],
    "d": ["c", "f"],
    "e": ["a", "b"],
    "f": ["e"],
  }
  np.random.seed(seed)
  metadata = [
    dict(zip(adjacency_dict.keys(), v))
    for v in random_metadata(len(adjacency_dict))
  ]
  return adjacency_dict, metadata


class TestDataMixin:
  """Mixin class for test cases that need example graph data"""
  def make_ids_fn(self):
    """Call :func:`conversion.make_ids_txt`"""
    return conversion.make_ids_txt(
        self.path("ids", "txt"),
        self.adjacency_path,
        lambda x: x,
    )

  def make_metadata_fn(self):
    """Call :func:`conversion.make_metadata_txt`"""
    return conversion.make_metadata_txt(
      self.path(),
      self.metadata_path,
      self.path("ids", "txt"),
      lambda x: x,
    )

  def setup_pickles_fn(self, seed: int = 42):
    """Setup pickle files and directories

    Args:
      seed (int): RNG seed"""
    self.adjacency_dict, self.metadata = test_data(seed)
    self.nnodes = len(self.adjacency_dict)
    self.path = pathutils.derived_paths(self.base_path)
    pickle.save_pickled(self.adjacency_dict, self.adjacency_path)
    pickle.save_pickled(self.metadata, self.metadata_path)
    os.makedirs(os.path.dirname(self.base_path), exist_ok=True)

  @contextlib.contextmanager
  def test_data(self, seed: int = 42):
    """Context manager for using test data on a fake filesystem

    Args:
      seed (int): RNG seed"""
    with fake_filesystem_unittest.Patcher():
      self.setup_pickles_fn(seed)
      yield
