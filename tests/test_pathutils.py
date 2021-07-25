"""Test utilities for handling file paths"""
from chromatictools import unitdoctest
from featgraph import pathutils


class DocTestsPathutils(metaclass=unitdoctest.DocTestMeta):
  """Pathutils doctests"""
  _modules = (pathutils,)
