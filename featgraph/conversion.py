"""Conversion functions for generating human-readable and BVGraph files
from the original pickled dataset"""
import os
import sys
import json
import pickle
import importlib
import itertools
import functools
import sortedcontainers
from chromatictools import cli
from featgraph import pathutils, logger, jwebgraph, scriptutils
from typing import Optional, Callable, Iterable, Tuple, Union, List


def make_ids_txt(dst: str,
                 src: str,
                 it: Optional[Callable[[Iterable], Iterable]] = None,
                 encoding="utf-8",
                 overwrite: bool = False) -> int:
  """Write the text file of artist ids

  Args:
    dst (str): Destination text file
    src (str): Source pickle file
    it (callable): Iterator wrapper function.
      If not :data:`None` the adjacency lists iterator will be wrapped using
      this function. Mainly intended for use with :data:`tqdm`
    encoding: Encoding for text files. Default is :data:`"utf-8"`
    overwrite (bool): If :data:`True`, then overwrite existing destination file

  Returns:
    int: The number of nodes"""
  if overwrite or pathutils.notisfile(dst):
    with open(dst, "w", encoding=encoding) as fout:
      with open(src, "rb") as fin:
        logger.info("Loading pickle file: %s", src)
        adjacency_lists = pickle.load(fin)
        logger.info("Sorting keys")
        ids = sortedcontainers.SortedSet(
            itertools.chain(
                adjacency_lists.keys(),
                itertools.chain.from_iterable(adjacency_lists.values())))
        logger.info("Writing file: %s", dst)
        if it is not None:
          ids = it(ids)
        for k in ids:
          fout.write(k + "\n")
        return len(ids)
  else:
    with open(dst, "r", encoding="utf-8") as fout:
      return sum(1 for _ in fout)


metadata_labels: Tuple[str, ...] = (
    "popularity",
    "genre",
    "name",
    "type",
    "followers",
)

metadata_fmt = {"genre": json.dumps}

metadata_missing = {"genre": []}


def make_metadata_txt(
    dst: Union[str, Callable],
    src: str,
    idf: str,
    it: Optional[Callable[[Iterable], Iterable]] = None,
    labels: Optional[Iterable[str]] = None,
    ext: str = ".txt",
    encoding="utf-8",
    overwrite: bool = False,
) -> List[str]:
  """Write the metadata text files

  Args:
    dst (str): Destination text file basepath
    src (str): Source pickle file
    idf (str): Graph node ids text filepath
    it (callable): Iterator wrapper function. If not :data:`None`
      the adjacency lists iterator will be wrapped using this function.
      Mainly intended for use with :data:`tqdm`
    labels (iterable of str): Labels for which to write a file.
      If :data:`None` (default), then write all metadata files
    ext (str): Common file extension. Default is :data:`".txt"`
    encoding: Encoding for output files. Default is :data:`"utf-8"`
    overwrite (bool): If :data:`True`, then overwrite existing destination file

  Returns:
    list of str: Output file paths"""
  written = []
  if not callable(dst):
    dst = pathutils.derived_paths(dst)
  with open(src, "rb") as f:
    logger.info("Loading pickle file: %s", src)
    metadata = pickle.load(f)
    if labels is None:
      labels = metadata_labels
    for k in labels:
      fmt = metadata_fmt.get(k, str)
      missing = metadata_missing.get(k, "")
      i = metadata_labels.index(k)
      fname = dst(k) + ext
      written.append(fname)
      if overwrite or pathutils.notisfile(fname):
        logger.info("Writing %s", fname)
        with open(fname, "w", encoding=encoding) as txt:
          with open(idf, "r", encoding=encoding) as ids:
            ids = (r.rstrip("\n") for r in ids)
            if it is not None:
              ids = it(ids)
            for a_id in ids:
              txt.write(fmt(metadata[i].get(a_id, missing)) + "\n")
  return written


def make_asciigraph_txt(
    dst: str,
    src: str,
    idf: str,
    it: Optional[Callable[[Iterable], Iterable]] = None,
    encoding="utf-8",
    overwrite: bool = False,
):
  """Write the text file of adjacency lists (ASCIIGraph)

  Args:
    dst (str): Destination text file path
    src (str): Source pickle file
    idf (str): Graph node ids text filepath
    it (callable): Iterator wrapper function. If not :data:`None`
      the adjacency lists iterator will be wrapped using this function.
      Mainly intended for use with :data:`tqdm`
    encoding: Encoding for output files. Default is :data:`"utf-8"`
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  if overwrite or pathutils.notisfile(dst):
    with open(idf, "r", encoding=encoding) as f:
      logger.info("Loading ids text file: %s", idf)
      ids = sortedcontainers.SortedSet(r.rstrip("\n") for r in f)
    with open(dst, "w", encoding=encoding) as txt:
      with open(src, "rb") as f:
        logger.info("Loading pickle file: %s", src)
        adjacency_lists = pickle.load(f)
        logger.info("Writing ASCIIGraph file: %s", dst)
        it = itertools.chain([len(ids)], iter(ids if it is None else it(ids)))
        for a_id in it:
          if isinstance(a_id, int):
            txt.write(str(a_id) + "\n")
            continue
          neighbors = map(ids.index, adjacency_lists.get(a_id, []))
          txt.write(" ".join(map(str, sorted(neighbors))) + "\n")


def compress_to_bvgraph(
    dst: str,
    src: Optional[str] = None,
    overwrite: bool = False,
):
  """Compress a text file of adjacency lists (ASCIIGraph) into a BVGraph

  Args:
    dst (str): Destination BVGraph file basepath
    src (str): Source text file basepath. If :data:`None`,
      then use the same basepath as :data:`dst`
    overwrite (bool): If :data:`True`,
      then overwrite existing destination file"""
  if src is None:
    src = dst
  srcpath = pathutils.derived_paths(src)
  dstpath = pathutils.derived_paths(dst)
  if overwrite or pathutils.notisfile(dstpath("graph")):
    webgraph = importlib.import_module("it.unimi.dsi.webgraph")
    logger.info("Loading ASCIIGraph: %s", srcpath("graph-txt"))
    ascii_graph = webgraph.ASCIIGraph.load(srcpath())
    logger.info("Compressing to BVGraph: %s", dstpath("graph"))
    webgraph.BVGraph.store(ascii_graph, dstpath())


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Run conversion script"""
  parser = scriptutils.FeatgraphArgParse(
      description="Convert original pickled dataset into text and BVGraph files"
  )
  parser.add_argument("adjacency_path",
                      help="The path of the adjacency lists pickle file")
  parser.add_argument("metadata_path",
                      help="The path of the metadata pickle file")
  parser.add_argument(
      "dest_path",
      help="The destination base path for the BVGraph and text files")
  args = parser.custom_parse(argv)

  # Make destination directory
  spotipath = pathutils.derived_paths(args.dest_path)
  spotidir = os.path.dirname(spotipath())
  if spotidir:
    os.makedirs(spotidir, exist_ok=True)
  # Make ids file
  nnodes = make_ids_txt(spotipath("ids", "txt"), args.adjacency_path, args.tqdm)
  # Make metadata files
  make_metadata_txt(
      spotipath,
      args.metadata_path,
      spotipath("ids", "txt"),
      args.tqdm if args.tqdm is None else functools.partial(args.tqdm,
                                                            total=nnodes),
  )
  # Make adjacency lists file
  make_asciigraph_txt(
      spotipath("graph-txt"),
      args.adjacency_path,
      spotipath("ids", "txt"),
      args.tqdm,
  )
  # Compress to BVGraph
  jwebgraph.jvm_process_run(
      compress_to_bvgraph,
      kwargs=dict(dst=spotipath(),),
      logging_kwargs=args.logging_kwargs,
      jvm_kwargs=dict(jvm_path=args.jvm_path,),
  )
