"""Functions for mapping subgenres to supergenres"""
import more_itertools
import functools
import json
import os
from typing import Dict, List, Optional, Iterable


_json_fname = ".".join((os.path.splitext(__file__)[0], "json"))


@functools.lru_cache(maxsize=1)
def get_default_map() -> Dict[str, List[str]]:
  """Get the default genre map dictionary"""
  with open(_json_fname, "r", encoding="utf-8") as f:
    d = json.load(f)
  return d


def supergenres(subgenre: str,
                genre_map: Optional[Dict[str, List[str]]] = None) -> List[str]:
  """Get the supergenres of a subgenre

  Args:
    subgenre (str): The name of the subgenre
    genre_map (dict): The subgenre-to-supergenres map dictionary.
      If :data:`None` (default), use the output of :func:`get_default_map`

  Returns:
    list of str: The list of supergenres names"""
  if genre_map is None:
    genre_map = get_default_map()
  return genre_map.get(subgenre, [])


def supergenres_from_iterable(
    subgenres: Iterable[str],
    genre_map: Optional[Dict[str, List[str]]] = None) -> List[str]:
  """Get the supergenres of a set of subgenres

  Args:
    subgenres (iterable of str): The names of the subgenres
    genre_map (dict): The subgenre-to-supergenres map dictionary.
      If :data:`None` (default), use the output of :func:`get_default_map`

  Returns:
    list of str: The list of supergenres names"""
  return list(
      set(
          more_itertools.flatten(
              map(functools.partial(supergenres, genre_map=genre_map),
                  subgenres))))
