"""Module for downloading and importing the WebGraph java library"""
import requests
from featgraph import logger
# --- JPype -------------------------------------------------------------------
# https://jpype.readthedocs.io/en/latest/quickguide.html
import jpype
import jpype.imports
# -----------------------------------------------------------------------------
import functools
import json
import os
from typing import Optional, Iterable, Dict


_json_file: str = "dependencies.json"


def download_to_file(
  file, url: str, chunk_size: Optional[int] = 1 << 13
) -> int:
  """Download file from URL

  Args:
    file: Writable stream
    url (str): URL of the file to download
    chunk_size (int): Chunk size for download stream

  Returns:
    int: Status code"""
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    for c in r.iter_content(chunk_size=chunk_size):
      file.write(c)
    return r.status_code


def path(*args, root: Optional[str] = None) -> str:
  """Get the path for a webgraph-related file

  Args:
    args: Path nodes
    root (str): Root folder for webgraph-related files.
      If :data:`None` (default), then the directory of this
      source file is the root

  Returns:
    str: Path"""
  if root is None:
    root = os.path.dirname(__file__)
  return os.path.join(root, *args)


def dependencies(file: Optional[str] = None) -> Dict[str, str]:
  """Get the dependency dictionary

  Args:
    file (str): The file path. If :data:`None` (default), then
      use default file path

  Returns:
    dict: Dependency dictionary where keys are file names and values are URLs"""
  filepath = path(_json_file) if file is None else file
  with open(filepath, "r") as fp:
    return json.load(fp)


def classpaths(
  root: Optional[str] = None,
  deps: Optional[Dict[str, str]] = None,
) -> Iterable[str]:
  """Get the classpath for all java library dependencies

  Args:
    root (str): Root for :func:`path`
    deps (dict): Dependency dictionary, as output by :func:`dependencies`.
      If :data:`None` (default), then get it from :func:`dependencies`

  Returns:
    iterable of str: The classpaths for the java library dependencies"""
  if deps is None:
    deps = dependencies()
  return map(
    functools.partial(path, root=root),
    deps.keys()
  )


def download_dependencies(
  deps: Optional[Dict[str, str]] = None,
  overwrite: bool = False,
  root: Optional[str] = None,
):
  """Download dependency files

  Args:
    deps (dict): Dependency dictionary, as output by :func:`dependencies`.
      If :data:`None` (default), then get it from :func:`dependencies`
    overwrite (bool): If :data:`True`, then overwrite existing files.
      Otherwise (default) skip download for existing files
    root (str): Root path for downloaded files. Argument for :func:`path`"""
  if deps is None:
    deps = dependencies()
  for k, v in deps.items():
    file_path = path(k, root=root)
    if os.path.isfile(file_path) and not overwrite:
      logger.info("Skipping download of %s. Found in %s", k, file_path)
    else:
      logger.info("Downloading %s from %s to %s", k, v, file_path)
      try:
        with open(file_path, "wb") as fp:
          download_to_file(fp, v)
      except Exception as e:
        if os.path.isfile(file_path):
          os.remove(file_path)
        raise e


def start_jvm(
  jvm_path: Optional[str] = None,
  download: bool = True,
  deps: Optional[Dict[str, str]] = None,
  root: Optional[str] = None,
  overwrite: bool = False,
):
  """Add jars to classpath and start the JVM

  Args:
    jvm_path (str): Path to a JVM executable
    download (bool): Whether to download dependency jars.
      Default is :data:`True`
    deps (dict): Dependency dictionary, as output by :func:`dependencies`.
      If :data:`None` (default), then get it from :func:`dependencies`
    root (str): Root path for downloaded files. Argument for :func:`path`
    overwrite (bool): If :data:`True`, then overwrite existing jar files.
      Otherwise (default) skip download for existing files"""
  if download:
    download_dependencies(
      deps=deps,
      overwrite=overwrite,
      root=root,
    )
  for cp in classpaths(root=root, deps=deps):
    jpype.addClassPath(cp)
  jpype.startJVM(*(() if jvm_path is None else (jvm_path,)))
