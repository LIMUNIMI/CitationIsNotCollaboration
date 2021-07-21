"""Conversion functions for generating human-readable and BVGraph files
from the original pickled dataset"""
import os
import sys
import pickle
import argparse
import importlib
from chromatictools import cli
from featgraph import pathutils, logger, jwebgraph
from typing import Optional, Callable, Iterable, Tuple, Union, List
import logging


def make_ids_txt(dst: str, src: str, it: Optional[Callable[[Iterable], Iterable]] = None, overwrite: bool = False):
  """Write the text file of artist ids

  Args:
    dst (str): Destination text file
    src (str): Source pickle file
    it (callable): Iterator wrapper function. If not :data:`None` the adjacency lists
      iterator will be wrapped using this function. Mainly intended for use with :data:`tqdm`
    overwrite (bool): If :data:`True`, then overwrite existing destination file"""
  if overwrite or pathutils.notisfile(dst):
    with open(dst, "w") as fout:
      with open(src, "rb") as fin:
        adjacency_lists = pickle.load(fin)
        if it is not None:
          adjacency_lists = it(adjacency_lists)
        for k in adjacency_lists:
          fout.write(k + "\n")


metadata_labels: Tuple[str, ...] = (
  "popularity",
  "genre",
  "name",
  "type",
  "followers",
)


def make_metadata_txt(
  dst: Union[str, Callable], src: str, idf: str,
  it: Optional[Callable[[Iterable], Iterable]] = None,
  labels: Optional[Iterable[str]] = None,
  ext: str = ".txt",
  encoding="utf-8",
  overwrite: bool = False,
  missing: str = "",
) -> List[str]:
  """Write the metadata text files

  Args:
    dst (str): Destination text file basepath
    src (str): Source pickle file
    idf (str): Graph node ids text filepath
    it (callable): Iterator wrapper function. If not :data:`None` the adjacency lists
      iterator will be wrapped using this function. Mainly intended for use with :data:`tqdm`
    labels (iterable of str): Labels for which to write a file. If :data:`None` (default),
      then write all metadata files
    ext (str): Common file extension. Default is :data:`".txt"`
    encoding: Encoding for output files. Default is :data:`"utf-8"`
    overwrite (bool): If :data:`True`, then overwrite existing destination file
    missing (str): String to write in place of missing values. Default is :data:`""`

  Returns:
    list of str: Output file paths"""
  written = []
  if not callable(dst):
    dst = pathutils.derived_paths(dst)
  with open(src, "rb") as f:
    metadata = pickle.load(f)
    if labels is None:
      labels = metadata_labels
    if it is not None:
      labels = it(labels)
    for k in labels:
      i = metadata_labels.index(k)
      fname = dst(k) + ext
      written.append(fname)
      if overwrite or pathutils.notisfile(fname):
        logger.debug("Writing %s", fname)
        with open(fname, "w", encoding=encoding) as txt:
          with open(idf, "r") as ids:
            for a_id in (r.rstrip("\n") for r in ids):
              txt.write(str(metadata[i].get(a_id, missing)) + "\n")
  return written


def make_adjacency_txt(
  dst: str, src: str, idf: str,
  it: Optional[Callable[[Iterable], Iterable]] = None,
  overwrite: bool = False,
  append: bool = True,
):
  """Write the text file of adjacency lists

  Args:
    dst (str): Destination text file path
    src (str): Source pickle file
    idf (str): Graph node ids text filepath
    it (callable): Iterator wrapper function. If not :data:`None` the adjacency lists
      iterator will be wrapped using this function. Mainly intended for use with :data:`tqdm`
    overwrite (bool): If :data:`True`, then overwrite existing destination file
    append (bool): If :data:`True`, then start writing at end of file"""
  if overwrite or append or pathutils.notisfile(dst):
    with open(idf, "r") as f:
      ids = list(map(lambda r: r.rstrip("\n"), f))
    if overwrite and os.path.exists(dst):
      os.remove(dst)
    with open(dst, "a+") as txt:
      txt.seek(0)
      it = iter(ids if it is None else it(ids))
      if append:
        for r, _ in zip(txt, it):
          pass  # skip written lines
      with open(src, "rb") as f:
        adjacency_lists = pickle.load(f)
        for a_id in it:
          # KeyError should not happen here
          # but better check for robustness.
          # It would happen only if there is
          # some inconsistency between the
          # adjacency lists pickle file and
          # the ids index text file
          neighbors = adjacency_lists.get(a_id, [])
          neighbors_i = []
          for n in neighbors:
            try:
              # ValueError can happen here
              # if successor has not an
              # associated entry in the
              # adjacency lists dict
              n_i = ids.index(n)
            except ValueError:
              continue
            else:
              neighbors_i.append(n_i)
          txt.write(" ".join(map(str, neighbors_i)) + "\n")


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Run conversion script"""
  parser = argparse.ArgumentParser(
    description="Convert original pickled dataset into text and BVGraph files"
  )
  parser.add_argument("adjacency_path", help="The path of the adjacency lists pickle file")
  parser.add_argument("metadata_path", help="The path of the metadata pickle file")
  parser.add_argument("dest_path", help="The destination base path for the BVGraph and text files")
  parser.add_argument(
    "--jvm-path", metavar="PATH",
    help="The Java virtual machine full path"
  )
  parser.add_argument(
    "-l", "--log-level", dest="log_level", metavar="LEVEL",
    default="INFO", type=lambda s: str(s).upper(),
    help="The logging level. Default is 'INFO'",
  )
  parser.add_argument(
    "--no-tqdm", dest="tqdm", action="store_false",
    help="Don't use tqdm progress bar",
  )
  args = parser.parse_args(argv)
  try:
    log_level = int(args.log_level)
  except ValueError:
    log_level = args.log_level
  logging.basicConfig(
    level=log_level,
    format='%(asctime)s %(name)-12s %(levelname)-8s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
  )
  logger.info(args)
  if args.tqdm:
    tqdm = importlib.import_module("tqdm").tqdm
  else:
    def tqdm(x, **kwargs):
      return x

  jwebgraph.start_jvm(jvm_path=args.jvm_path)
  # Make destination directory
  spotipath = pathutils.derived_paths(args.dest_path)
  os.makedirs(os.path.dirname(spotipath()), exist_ok=True)
  # Make ids file
  make_ids_txt(
    spotipath("ids", "txt"),
    args.adjacency_path,
    tqdm
  )
  # Make metadata files
  make_metadata_txt(
    spotipath,
    args.metadata_path,
    spotipath("ids", "txt"),
    tqdm,
  )
  # Make adjacency lists file
  make_adjacency_txt(
    spotipath("adj", "txt"),
    args.adjacency_path,
    spotipath("ids", "txt"),
    tqdm,
  )
