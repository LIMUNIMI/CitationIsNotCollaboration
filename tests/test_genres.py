"""Test genre utilities"""
import unittest
import more_itertools
from featgraph import genre_map
import numpy as np


class TestGenres(unittest.TestCase):
  """Test genre utilities"""

  def test_json_data(self):
    """Test that the loaded json is of the correct type"""
    for k, v in genre_map.get_default_map().items():
      with self.subTest(subgenre=k):
        self.assertTrue(isinstance(k, str))
      with self.subTest(subgenre=k, supergenre=v):
        self.assertTrue(isinstance(v, list))
        for vi in v:
          self.assertTrue(isinstance(vi, str))

  def test_supergenres(self):
    """Test correctness of supergenres from json"""
    for k, v in genre_map.get_default_map().items():
      with self.subTest(subgenre=k, supergenre=v):
        self.assertEqual(genre_map.supergenres(k), v)
    with self.subTest(subgenre="inexistent"):
      self.assertEqual(genre_map.supergenres("go1 yv'p8"), [])

  def test_supergenres_it(self):
    """Test correctness of supergenres from iterable"""
    np.random.seed(42)
    subgenres = np.random.choice(list(genre_map.get_default_map().keys()),
                                 (20, 3))
    for s in subgenres:
      supergenres = genre_map.supergenres_from_iterable(s)
      with self.subTest(subgenres=s, supergenres=supergenres):
        for t in more_itertools.flatten(map(genre_map.supergenres, s)):
          self.assertIn(t, supergenres)
