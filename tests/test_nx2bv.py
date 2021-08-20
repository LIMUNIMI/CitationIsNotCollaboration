"""Test networkx to BVGraph conversion functions"""
from featgraph import sgc, nx2bv, pathutils
from tests import testutils
import unittest
import os


class TestNx2Bv(unittest.TestCase):
  """Test networkx to BVGraph conversion functions"""
  @classmethod
  def setUpClass(cls):
    cls.base_path = "gRaphZ_nx2bv/testExampLe-6921"
    cls.seed = 42
    cls.model = sgc.SGCModel()
    cls.nxgraph = cls.model(seed=cls.seed)

  def test_whole_pipeline(self):
    """Test whole pipeline"""
    path = pathutils.derived_paths(
      os.path.join(".tmp_nx2BV_test", self.base_path)
    )

    nx2bv.nx2bv(self.nxgraph, path())
    # check neighbors from asciigraph
    for n, nbrdict in self.nxgraph.adjacency():
      with self.subTest(check="neighbors", file="asciigraph", node=n):
        self.assertTrue(testutils.check_neighbors(
          path(), n, list(nbrdict.keys()), attr="index",
        ))

    # delete asciigraph
    s = ("graph-txt",)
    with self.subTest(check_exists=s):
      self.assertTrue(os.path.isfile(path(*s)))
    os.remove(path(*s))
    os.rmdir(os.path.dirname(path()))
    os.rmdir(os.path.dirname(os.path.dirname(path())))
