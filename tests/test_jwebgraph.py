"""Test WebGraph java library correctly loading"""
from featgraph import jwebgraph, metadata
from tests import testutils
from unittest import mock
import numpy as np
import importlib
import unittest
import requests
import os
import io


def test_import_webgraph(name: str, mod: str = "it.unimi.dsi.webgraph") -> bool:
  """Test that the jvm starts correctly and jar can be imported.
  This needs to be done in a separate thread to make sure that
  the JVM properly shuts down"""
  cls = getattr(importlib.import_module(mod), name)
  return cls.__name__ == ".".join((mod, name))


class TestJWebGraph(unittest.TestCase):
  """Tests related to WebGraph with JPype"""
  def test_download_to_file(self):
    """Test that binary files get downloaded correctly"""
    url = "https://gist.githubusercontent.com/ChromaticIsobar/" \
          "ccb3ea9aa45190fd1f5313c79eb9333f/raw/" \
          "30987b9895c2ae2cc6dbd1cfa76db24bf2981fac/lipsum.txt"
    content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
    with io.BytesIO() as buf:
      jwebgraph.download_to_file(buf, url)
      buf.seek(0)
      rcontent = buf.read().decode("ascii")
      self.assertEqual(content, rcontent)

  def test_download_jars(self):
    """Test that all jars get downloaded. Check overwrite behaviour, too"""
    tmp_root = os.path.abspath(".tmp_clAsSPatTh_dload")
    os.makedirs(tmp_root, exist_ok=True)
    jwebgraph.download_dependencies(root=tmp_root)
    n = 0
    for cp in jwebgraph.classpaths(root=tmp_root):
      n += 1
      with self.subTest(classpath=cp):
        self.assertTrue(os.path.isfile(cp))
    with mock.patch.object(jwebgraph, "download_to_file") as dtf:
      jwebgraph.download_dependencies(root=tmp_root)
      with self.subTest(overwrite=False):
        self.assertEqual(dtf.call_count, 0)
      jwebgraph.download_dependencies(root=tmp_root, overwrite=True)
      with self.subTest(overwrite=True):
        self.assertEqual(dtf.call_count, n)
    for cp in jwebgraph.classpaths(root=tmp_root):
      os.remove(cp)
    os.rmdir(tmp_root)

  def test_download_fail(self):
    """Test that temporary file is cleaned up on a download fail"""
    tmp_root = os.path.abspath(".tmp_clAsSPatTh_dloadFAeel")
    deps = {
      "ancient.jar": "https://i.cannot.find/this/ancient/artifact/ancient.jar",
    }
    os.makedirs(tmp_root, exist_ok=True)
    with self.subTest(step="exception"):
      with self.assertRaises(requests.exceptions.RequestException):
        jwebgraph.download_dependencies(deps=deps, root=tmp_root)
    for k in deps:
      with self.subTest(step="cleanup", what=k):
        self.assertFalse(os.path.isfile(
          jwebgraph.path(k, root=tmp_root)
        ))
    os.rmdir(tmp_root)

  def test_start_jvm(self):
    """Test that the jvm starts correctly and jar can be imported"""
    tmp_root = os.path.abspath(".tmp_clAsSPatTh_jvm")
    os.makedirs(tmp_root, exist_ok=True)

    b = jwebgraph.jvm_process_run(
      test_import_webgraph,
      args=("BVGraph",),
      jvm_kwargs=dict(
        jvm_path=testutils.jvm_path,
        root=tmp_root,
      ),
      return_type="B",
    )
    self.assertEqual(b, 1)

    for cp in jwebgraph.classpaths(root=tmp_root):
      os.remove(cp)
    os.rmdir(tmp_root)


def test_jaccard(a, b):
  """Test Jaccard index"""
  return importlib.import_module(
    "featgraph.jwebgraph.utils"
  ).jaccard(a, b)


def test_load_as_doubles(
  fname: str, n: int = 128, scale: float = 1024, seed: int = 42
) -> int:
  """Test loadAsDoubles"""
  np.random.seed(seed)
  a = (np.random.rand(n) * scale).astype(int)
  with open(fname, "w") as f:
    for x in a:
      f.write("{:.0f}\n".format(x))
  b = importlib.import_module(
    "featgraph.jwebgraph.utils"
  ).load_as_doubles(fname)
  return sum(x != int(y) for x, y in zip(a, b))


class TestUtils(
  testutils.TestDataMixin,
  unittest.TestCase
):
  """Test JWebGraph utils"""
  def test_module_not_found(self):
    """Test error if import with no jvm"""
    with self.assertRaises(ModuleNotFoundError):
      importlib.import_module("featgraph.jwebgraph.utils")

  def test_jaccard(self):
    """Test Jaccard index"""
    path = "graPh-fAkePaTH"
    for union in range(1, 4):
      for intersection in range(union + 1):
        na = (union - intersection) // 2 + intersection
        a = [metadata.Artist(path, index=i) for i in range(na)]
        b = [
          metadata.Artist(path, index=i)
          for i in range(na - intersection, union)
        ]
        with self.subTest(union=union, intersection=intersection):
          self.assertEqual(
            intersection/union,
            jwebgraph.jvm_process_run(
              test_jaccard,
              jvm_kwargs=dict(jvm_path=testutils.jvm_path),
              return_type="d",
              args=(a, b),
            )
          )

  def test_load_as_doubles(self):
    """Test loadAsDoubles"""
    fname = ".tmp_load_As_dOuBl3z.txt"
    self.assertEqual(0, jwebgraph.jvm_process_run(
      test_load_as_doubles,
      jvm_kwargs=dict(jvm_path=testutils.jvm_path),
      return_type="I",
      kwargs=dict(fname=fname)
    ))
    os.remove(fname)
