"""Miscellaneous functions and classes"""
from typing import Union, Callable, Sequence


VectorOrCallable = Union[Callable[[], Sequence], Sequence]


def pretty_print_int(n: int, k: int = 3, sep: str = " ") -> str:
  """Print an integer with thousands separators

  Args:
    n (int): Integer to print
    k (int): Interval for separators. Default is :data:`3` (for thousands)
    sep (str): Separator. Default is :data:`" "`

  Returns:
    str: Pretty string"""
  def _ppi_it(s: str):
    i = 0
    for c in reversed(s):
      if i == k:
        i = 0
        yield sep
      yield c
      i += 1
  return "".join(reversed(list(_ppi_it(str(n)))))


def jaccard(a: Sequence, b: Sequence) -> float:
  """Jaccard index between sets

  Args:
    a: First set of values
    b: Second set of values

  Returns:
    float: Jaccard index"""
  i = len(set(a).intersection(b))
  return i / (len(a) + len(b) - i)
