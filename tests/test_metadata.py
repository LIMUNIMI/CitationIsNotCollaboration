"""Test metadata wrapper"""
from featgraph import metadata, conversion
from tests import testutils
import unittest


class TestMetadata(
  testutils.TestDataMixin,
  unittest.TestCase
):
  """Test metadata wrapper"""
  adjacency_path = "fAkeDATa_metadata/collaboration_network_edge_list.pickle"
  metadata_path = "fAkeDATa_metadata/artist_data.pickle"
  base_path = "grAp4s_metadata/testexampLe-6569"

  def test_metadata_handle(self):
    """Test reading metadata"""
    with self.test_data():
      self.make_ids_fn()
      self.make_metadata_fn()
      conversion.make_asciigraph_txt(
        self.path("graph-txt"), self.adjacency_path, self.path("ids", "txt"),
      )
      for aid in self.adjacency_dict:
        a = metadata.Artist(self.path(), aid=aid)
        with self.subTest(property="index", aid=aid):
          self.assertEqual(a.index, ord(aid) - ord("a"))
        with self.subTest(property="aid", aid=aid):
          self.assertEqual(a.aid, aid)
        with self.subTest(property="name", aid=aid):
          self.assertTrue(isinstance(a.name, str))
        with self.subTest(property="followers", aid=aid):
          self.assertTrue(isinstance(a.followers, int))
        with self.subTest(property="neighbors", aid=aid):
          self.assertTrue(isinstance(a.neighbors, list))
          for n in a.neighbors:
            self.assertTrue(isinstance(n, metadata.Artist))
        with self.subTest(property="genre", aid=aid):
          self.assertTrue(isinstance(a.genre, list))
          for g in a.genre:  # pylint: disable=E1133 (false positive)
            self.assertTrue(isinstance(g, str))
        with self.subTest(property="degree", aid=aid):
          self.assertEqual(a.degree, len(a.neighbors))
        with self.subTest(property="popularity", aid=aid):
          self.assertTrue(isinstance(a.popularity, float))
        with self.subTest(property="type", aid=aid):
          self.assertEqual(a.type, "artist")
        with self.subTest(property="__str__", aid=aid):
          self.assertTrue(str(a).startswith("Artist(aid="))

  def test_noinput_error(self):
    """Test error on no specification"""
    with self.test_data():
      self.make_ids_fn()
      self.make_metadata_fn()
      with self.assertRaises(ValueError):
        metadata.Artist(self.path()).index  # pylint: disable=W0106

  def test_miss_error(self):
    """Test error on artist ID miss"""
    with self.test_data():
      self.make_ids_fn()
      self.make_metadata_fn()
      with self.assertRaises(ValueError):
        metadata.Artist(self.path(), aid="01234567890").index  # pylint: disable=W0106

  def test_index_provided(self):
    """Test that the index is returned correctly"""
    with self.test_data():
      self.make_ids_fn()
      self.make_metadata_fn()
      for i in range(self.nnodes):
        with self.subTest(index=i):
          self.assertEqual(
            i, metadata.Artist(self.path(), index=i).index
          )

  def test_index_outofrange(self):
    """Test that the index out of range causes error"""
    with self.test_data():
      self.make_ids_fn()
      self.make_metadata_fn()
      with self.assertRaises(EOFError):
        metadata.Artist(self.path(), index=self.nnodes).aid  # pylint: disable=W0106
