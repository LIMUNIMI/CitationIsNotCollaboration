"""Test networkx to BVGraph conversion functions"""
from featgraph import sgc, nx2bv, pathutils, jwebgraph
from tests import testutils
import unittest
import os


class TestNx2Bv(
  testutils.TestDataMixin,
  unittest.TestCase
):
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
    with self.check_files_exist(
      os.path.dirname(path()), os.path.dirname(os.path.dirname(path())),
    ):
      with self.check_files_exist(path("graph-txt")):
        # Convert
        jwebgraph.jvm_process_run(
          nx2bv.nx2bv,
          kwargs=dict(
            graph=self.nxgraph,
            bvgraph_basepath=path(),
            overwrite=True,
          ),
          jvm_kwargs=dict(
            jvm_path=testutils.jvm_path,
          ),
        )

        # check neighbors from asciigraph
        for n, nbrdict in self.nxgraph.adjacency():
          with self.subTest(check="neighbors", file="asciigraph", node=n):
            self.assertTrue(testutils.check_neighbors(
              path(), n, list(nbrdict.keys()), attr="index",
            ))
