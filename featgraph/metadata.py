"""Convenience classes for accessing artist metadata from text files"""
from featgraph import pathutils
import featgraph.misc
import more_itertools
import functools
import json
import importlib
from typing import Optional, Callable, Sequence, Dict, Tuple, List


class Artist:
  """Wrapper class for inspecting artist data and metadata.
  Initialization requires a base path for the files and at least one of:
  index, artist ID, or name

  Args:
    basepath (str): Base path for graph files
    index (int): Artist index in files (starting from 0)
    aid (str): Artist ID
    name (str): Artist name
    sep (str): Suffix separator. Defaults to :data:`"."`
    followers_file_suffix (sequence of str): Suffix for followers file
    genre_file_suffix (sequence of str): Suffix for genre file
    adj_file_suffix (sequence of str): Suffix for ASCIIGraph file
    id_file_suffix (sequence of str): Suffix for artist id file
    name_file_suffix (sequence of str): Suffix for name file
    popularity_file_suffix (sequence of str): Suffix for popularity file
    type_file_suffix (sequence of str): Suffix for type file
    encoding: File encoding. Defaults to :data:`"utf-8"`"""
  def __init__(
    self, basepath: str,
    index: Optional[int] = None,
    aid: Optional[str] = None,
    name: Optional[str] = None,
    sep: str = ".",
    followers_file_suffix: Sequence[str] = ("followers", "txt"),
    genre_file_suffix: Sequence[str] = ("genre", "txt"),
    adj_file_suffix: Sequence[str] = ("graph-txt",),
    id_file_suffix: Sequence[str] = ("ids", "txt"),
    name_file_suffix: Sequence[str] = ("name", "txt"),
    popularity_file_suffix: Sequence[str] = ("popularity", "txt"),
    type_file_suffix: Sequence[str] = ("type", "txt"),
    encoding="utf-8",
  ):
    self.basepath = basepath
    self.sep = sep
    self.followers_file_suffix = followers_file_suffix
    self.genre_file_suffix = genre_file_suffix
    self.adj_file_suffix = adj_file_suffix
    self.id_file_suffix = id_file_suffix
    self.name_file_suffix = name_file_suffix
    self.popularity_file_suffix = popularity_file_suffix
    self.type_file_suffix = type_file_suffix
    self.encoding = encoding

    self._index = index
    self._aid = aid
    self._name = name

  def __str__(self) -> str:
    """Display artist ID and name (if name is in metadata)"""
    name = "" if self.name is None else ", name='{}'".format(self.name)
    return "Artist(aid='{}'{})".format(self.aid, name)

  @property
  def _derived_paths(self) -> Callable[[str], str]:
    """Get paths stemming from the base path"""
    return pathutils.derived_paths(self.basepath, self.sep)

  @property
  def _filedict(self) -> Dict[str, Tuple[str, Sequence[str], int]]:
    """Dictionary of files metadata. Keys are metadata labels and values are:

    - user-provided value
    - file suffix
    - file offset"""
    return {
      "followers": (None, self.followers_file_suffix, 0),
      "genre": (None, self.genre_file_suffix, 0),
      "neighbors": (None, self.adj_file_suffix, -1),
      "aid": (self._aid, self.id_file_suffix, 0),
      "name": (self._name, self.name_file_suffix, 0),
      "popularity": (None, self.popularity_file_suffix, 0),
      "type": (None, self.type_file_suffix, 0),
    }

  @property
  @functools.lru_cache(maxsize=1)
  def index(self) -> int:
    """Artist index in files (starting from 0)"""
    if self._index is not None:
      return self._index
    for k, (v, suffix, offset) in self._filedict.items():
      if v is None:
        continue
      fname = self._derived_paths(*suffix)
      with open(fname, "r", encoding=self.encoding) as f:
        for i, n in enumerate(r.rstrip("\n") for r in f):
          if n == v:
            return i + offset
      raise ValueError("No artist was found for {} '{}' in file '{}'".format(
        k, v, fname
      ))
    raise ValueError("Please, specify at least one of: index, aid, name")

  def _property_from_file(self, kp: str) -> str:
    """Get property value from file

    Args:
      kp (str): Property label

    Returns:
      str: The i-th row in the metadata file, where i is the artist's index"""
    vp, suffix, offset = self._filedict[kp]
    if vp is not None:
      return vp
    fname = self._derived_paths(*suffix)
    with open(fname, "r", encoding=self.encoding) as f:
      v = more_itertools.nth(f, self.index - offset)
      if v is None:
        raise EOFError(
          "Ran out of input while looking for row {} in file '{}'".format(
            self.index, fname
          )
        )
      return v.rstrip("\n")

  @property
  @functools.lru_cache(maxsize=1)
  def followers(self) -> Optional[int]:
    """Number of followers of the artist. Defaults to :data:`None`
      if value is missing"""
    s = self._property_from_file("followers")
    return int(s) if s else None

  @property
  @functools.lru_cache(maxsize=1)
  def genre(self) -> Optional[Sequence[str]]:
    """Artist music genres. Defaults to :data:`None`
      if value is missing"""
    s = self._property_from_file("genre")
    return json.loads(s.replace("'", "\"")) if s else None

  @property
  @functools.lru_cache(maxsize=1)
  def neighbors(self) -> List["Artist"]:
    """Neighbors of the artist in the graph"""
    try:
      # default to ASCIIGraph file
      indices = self._property_from_file("neighbors").split(" ")
    except FileNotFoundError:
      # fall back to BVGraph file
      indices = featgraph.misc.IteratorWrapper(
        importlib.import_module(
          "featgraph.jwebgraph.utils"
        ).BVGraph(self.basepath).load().successors(self.index),
        "nextInt", -1
      )
    return [
      Artist(basepath=self.basepath, index=int(i))
      for i in indices if i
    ]

  @property
  @functools.lru_cache(maxsize=1)
  def degree(self) -> int:
    """Out-degree of the artist in the graph"""
    return len(self.neighbors)

  @property
  @functools.lru_cache(maxsize=1)
  def aid(self) -> str:
    """Artist ID"""
    return self._property_from_file("aid")

  @property
  @functools.lru_cache(maxsize=1)
  def name(self) -> Optional[str]:
    """Artist name. Defaults to :data:`None` if value is missing"""
    return self._property_from_file("name") or None

  @property
  @functools.lru_cache(maxsize=1)
  def popularity(self) -> Optional[float]:
    """Artist popularity as the number of streams as a percentage of the most
      popular artist's streams (in January 2018 data, Ed Sheeran). Defaults to
      :data:`None` if value is missing"""
    s = self._property_from_file("popularity")
    return float(s) if s else None

  @property
  @functools.lru_cache(maxsize=1)
  def type(self) -> Optional[str]:
    """Type of content creator (e.g. :data:`"artist"`).
      Defaults to :data:`None` if value is missing"""
    return self._property_from_file("type") or None
