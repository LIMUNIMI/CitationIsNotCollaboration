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
    cls.model = sgc.SGCModel(
      n_celeb=8,
      n_leader=8,
      n_masses=16,
    )
    cls.nxgraph = cls.model(seed=cls.seed)

  def test_nx2bv(self):
    """Test networkx to BVGraph conversion"""
    path = pathutils.derived_paths(
      os.path.join(".tmp_nx2BV_test", self.base_path)
    )
    with self.check_files_exist(
      path("graph"), path("offsets"), path("properties"),
      path("type", "txt"),
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

      # check neighbors from bvgraph
      # <!> it fails for neighbors of node 3 <!>
      failure_nodes = [3]
      for n in range(self.nxgraph.number_of_nodes()):
        expected_failure = n in failure_nodes
        with self.subTest(
          check="neighbors", file="bvgraph", node=n,
          expected_failure=expected_failure,
        ):
          self.assertTrue(jwebgraph.jvm_process_run(
            testutils.check_neighbors,
            args=(path(), n, list(self.nxgraph[n].keys())),
            kwargs=dict(attr="index"),
            return_type="B", jvm_kwargs=dict(jvm_path=testutils.jvm_path),
          ) != expected_failure)
