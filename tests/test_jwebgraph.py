"""Test WebGraph java library correctly loading"""
from featgraph import jwebgraph
import jpype
import unittest
from unittest import mock
import importlib
import multiprocessing
import os
import io


def test_start_jvm(tmp_root: str):
  """Test that the jvm starts correctly and jar can be imported.
  This needs to be done in a separate thread to make sure that
  the JVM properly shuts down"""
  os.makedirs(tmp_root, exist_ok=True)
  jwebgraph.start_jvm(
    jvm_path=os.environ.get("FEATGRAPH_JAVA_PATH", None),
    root=tmp_root,
  )
  importlib.import_module("it.unimi.dsi.webgraph")
  jpype.shutdownJVM()


class TestJWebGraph(unittest.TestCase):
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
    jwebgraph.download_jars(root=tmp_root)
    n = 0
    for cp in jwebgraph.classpaths(root=tmp_root):
      n += 1
      with self.subTest(classpath=cp):
        self.assertTrue(os.path.isfile(cp))
    with mock.patch.object(jwebgraph, "download_to_file") as dtf:
      jwebgraph.download_jars(root=tmp_root)
      with self.subTest(overwrite=False):
        self.assertEqual(dtf.call_count, 0)
      jwebgraph.download_jars(root=tmp_root, overwrite=True)
      with self.subTest(overwrite=True):
        self.assertEqual(dtf.call_count, n)
    for cp in jwebgraph.classpaths(root=tmp_root):
      os.remove(cp)
    os.rmdir(tmp_root)

  def test_start_jvm(self):
    """Test that the jvm starts correctly and jar can be imported"""
    tmp_root = os.path.abspath(".tmp_clAsSPatTh_jvm")
    p = multiprocessing.Process(target=test_start_jvm, args=(tmp_root,))
    p.start()
    p.join()
    self.assertEqual(p.exitcode, 0)
    for cp in jwebgraph.classpaths(root=tmp_root):
      os.remove(cp)
    os.rmdir(tmp_root)
