"""Tests for the misc module"""
import featgraph.misc
import itertools
import unittest


class TestMisc(unittest.TestCase):
  """Tests for the misc module"""

  def test_ppi(self):
    """Test pretty-print int"""
    d = ("0", "100", "1 250", "500 000", "1 750 321", "- 100", "-3 249")
    for k in d:
      v = int(k.replace(" ", ""))
      with self.subTest(n=k):
        self.assertEqual(k, featgraph.misc.pretty_print_int(v))

  def test_jaccard(self):
    """Test Jaccard index"""
    for union in range(1, 6):
      for intersection in range(union + 1):
        na = (union - intersection) // 2 + intersection
        a = list(range(na))
        b = list(range(na - intersection, union))
        with self.subTest(union=union, intersection=intersection):
          self.assertEqual(intersection / union, featgraph.misc.jaccard(a, b))

  def test_iterator_wrapper_si(self):
    """Test IteratorWrapper on StopIteration"""
    self.assertEqual(
        list(featgraph.misc.IteratorWrapper(iter(range(5)))),
        list(range(5)),
    )

  def test_iterator_wrapper_ev(self):
    """Test IteratorWrapper on end_value"""
    self.assertEqual(
        list(
            featgraph.misc.IteratorWrapper(
                iter(itertools.chain(range(5), itertools.repeat(None))))),
        list(range(5)),
    )
