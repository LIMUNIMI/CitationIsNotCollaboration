"""Miscellaneous functions and classes"""
import pandas as pd
import contextlib
import functools
from typing import Union, Callable, Sequence, Iterator, ContextManager, Optional

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


class IteratorWrapper:
  """Wrapper for an iterator. Mainly intended for wrapping Java iterators

  Args:
    it: Iterator
    next_method (str): Name of the method used to iterate one step
    end_value: Stop iteration when this value is found"""

  def __init__(self, it, next_method: str = "__next__", end_value=None):
    self.it = it
    self.next_method = next_method
    self.end_value = end_value

  def __iter__(self):
    """Start iteration (does nothing)"""
    return self

  def __next__(self):
    """Iterate one step. Stop when end value is returned"""
    v = getattr(self.it, self.next_method)()
    if v == self.end_value:
      raise StopIteration()
    return v


class NodeIteratorWrapper(IteratorWrapper):
  """Wrapper for a node iterator

  Args:
    it: NodeIterator"""

  def __init__(self, it, next_method: str = "nextInt", end_value=-1):
    super().__init__(it, next_method=next_method, end_value=end_value)


@contextlib.contextmanager
def multicontext(it: Iterator[ContextManager]):
  """Context manager wrapper for multiple contexts managers

  Args:
    it: Iterator of context managers to wrap

  Yields:
    tuple: The tuple of values yielded by the individual context managers"""
  try:
    cm = next(it)
  except StopIteration:
    yield ()
  else:
    with cm as value:
      with multicontext(it) as values:
        yield (value, *values)


def sorted_values(df: pd.DataFrame,
                  key: str = "mean",
                  norm: Optional[str] = None,
                  sort_key: str = "threshold",
                  **kwargs):
  """Get the values of a dataframe column filtering
  and sorting by other columns

  Args:
    df (DataFrame): The dataframe to process
    key (str): The column name of the values to return
    norm (str): If not :data:`None`, then normalize values by this column
    sort_key (str): The name of the column to use for sorting
    kwargs: Key-value pairs. Only the rows for which the column with the key
      has the specified values are returned

    Returns:
      array: Sorted values"""
  df_ = df[functools.reduce(
      lambda x, y: x & y,
      map(
          lambda t: df[t[0]] == t[1],
          kwargs.items(),
      ),
  )].copy()
  df_.sort_values(sort_key, inplace=True)
  df_.reset_index(inplace=True)
  a = df_[key]
  if norm is not None:
    a = a / df_[norm]
  return a
