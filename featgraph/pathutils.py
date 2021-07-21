"""Utilities for handling file paths"""
import os
import glob
from typing import Callable, Union, Optional
import logging


def notisfile(f: str, func: Union[Callable, bool] = os.path.isfile, msg: str = "Found '%s'. Skipping"):
  """Check if file does not exist. If file is found, log a message

  Args:
    f (str): File path to check
    func (bool or callable): If a boolean, then use this as the truth value for the file existence.
      Otherwise, call this function to get the truth value. Defaults to :func:`os.path.isfile`
    msg (str): Log message. It should have a string formattable field where the file name will be interpolated

  Example:
    >>> from featgraph.pathutils import notisfile
    >>> notisfile("inexistent.5619hf")
    True
    >>> notisfile("inexistent.5619hf", True)
    False"""
  if isinstance(func, bool):
    b = func
  else:
    b = func(f)
  if b:
    logging.info(msg, f)
  return not b


def notisglob(f: str, func: Union[Callable, bool] = len, **kwargs):
  """Check if no file that matches the expression exists. If a file is found, log a message

  Args:
    f (str): File pattern to check
    func (bool or callable): If a boolean, then use this as the truth value for the file existence.
      Otherwise, call this function to get the truth value. Defaults to :func:`len`
    kwargs: Keyword arguments for :func:`notisfile`

  Example:
    >>> from featgraph.pathutils import notisglob
    >>> notisglob("/path/to/nothing/*.ivsiva")
    True"""
  return notisfile(f=glob.glob(f), func=func, **kwargs)


def derived_paths(f: str, sep: str = ".") -> Callable[[str], str]:
  """Returns a helper function for generating derived paths

  Args:
    f (str): Base path
    sep (str): Separator between base path and extension(s)

  Returns:
    Helper function

  Example:
    >>> from featgraph.pathutils import derived_paths
    >>> dbpath = derived_paths("db")
    >>> dbpath()
    'db'
    >>> dbpath("csv")
    'db.csv'
    >>> dbpath("readme", "md")
    'db.readme.md'"""
  def derived_paths_(*suffix: str) -> str:
    if len(suffix) < 1:
      return f
    return sep.join((f, *suffix))
  return derived_paths_
