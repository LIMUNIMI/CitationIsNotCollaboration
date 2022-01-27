"""Utils for CLI scripts"""
from featgraph import jwebgraph, logger, pathutils
import collections
import numpy as np
import argparse
import logging
import importlib
import json
import os
from typing import Optional


class EchoDict(dict):
  """Dictionary that defaults to the key value on misses"""

  def __getitem__(self, k):
    try:
      return super().__getitem__(k)
    except KeyError:
      return k


class FeatgraphArgParse(argparse.ArgumentParser):
  """Argument parser for Featgraph CLI scripts

  Args:
    kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.add_argument("--jvm-path",
                      metavar="PATH",
                      help="The Java virtual machine full path")
    self.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        metavar="LEVEL",
        default="WARN",
        type=lambda s: str(s).upper(),
        help="The logging level. Default is 'WARN'",
    )
    self.add_argument(
        "--tqdm",
        action="store_true",
        help="Use tqdm progress bar (you should install tqdm for this)",
    )

  def custom_parse(self, argv):
    """Parse arguments, configure logger and eventually import :mod:`tqdm`"""
    args = self.parse_args(argv)
    try:
      log_level = int(args.log_level)
    except ValueError:
      log_level = args.log_level
    args.logging_kwargs = dict(
        level=log_level,
        format="%(asctime)s %(name)-12s %(levelname)-8s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.basicConfig(**args.logging_kwargs)
    args.tqdm = importlib.import_module("tqdm").tqdm if args.tqdm else None
    return args


class FeatgraphPlotsArgParse(FeatgraphArgParse):
  """Argument parser for plot scripts

  Args:
    style_file (str): File path to a default :data:`.mplstyle` file
    max_width (float): Default max width of plots
    max_height (float): Default max height of plots
    kwargs: Keyword arguments for :class:`argparse.ArgumentParser`"""

  def __init__(self,
               style_file: Optional[str] = None,
               abbrev_file: Optional[str] = None,
               palette_file: Optional[str] = None,
               max_width: float = 10.0,
               max_height: float = 10.0,
               **kwargs):
    super().__init__(**kwargs)

    self.add_argument("graph_path", help="The base path for the BVGraph files")
    self.add_argument("dest_path",
                      help="The destination path (directory) "
                      "for the plot files")
    self.add_argument("--csv-path",
                      default=None,
                      help="The path of the centralities summary dataframe")
    self.add_argument("-F",
                      "--force",
                      dest="overwrite",
                      action="store_true",
                      help="Overwrite preexisting files")
    self.add_argument("-a",
                      "--artist",
                      dest="ref_artists",
                      metavar="AID",
                      type=str,
                      action="append",
                      help="Specify a reference artist by ID")
    self.add_argument("--sgc-path",
                      metavar="PATH",
                      default=".sgc-graph/sgc",
                      help="The path for the SGC random graph")
    self.add_argument("--sgc-seed",
                      metavar="SEED",
                      default=42,
                      help="The seed for the SGC random graph")
    self.add_argument("--max-height",
                      metavar="H",
                      default=max_height,
                      type=float,
                      help="The maximum height of figures (in inches)")
    self.add_argument("--max-width",
                      metavar="W",
                      default=max_width,
                      type=float,
                      help="The maximum width of figures (in inches)")
    self.add_argument("--mpl-style",
                      metavar="FILEPATH",
                      default=style_file,
                      help="The path of the matplotlib style file")
    self.add_argument("--abbrev",
                      metavar="FILEPATH",
                      default=abbrev_file,
                      help="The path of the genre abbreviations JSON file")
    self.add_argument("--palette",
                      metavar="FILEPATH",
                      default=palette_file,
                      help="The path of the genre palette JSON file")
    self.add_argument("--palette-saturation",
                      metavar="S",
                      default=0.75,
                      help="Palette saturation multiplier")
    self.add_argument("--palette-alpha",
                      metavar="A",
                      default=0.85,
                      help="Palette opacity")

  @staticmethod
  def none_or_expand(s: Optional[str]) -> Optional[str]:
    return None if s is None else os.path.expanduser(s)

  def custom_parse(self, argv):
    """Parse and preprocess arguments, start JVM and load graph"""
    args = super().custom_parse(argv)

    args.jvm_path = self.none_or_expand(args.jvm_path)
    args.dest_path = self.none_or_expand(args.dest_path)
    args.graph_path = self.none_or_expand(args.graph_path)

    # Configure matplotlib
    plt = importlib.import_module("matplotlib.pyplot")
    mpl = importlib.import_module("matplotlib")
    logger.info("Setting style: %s", args.mpl_style)
    plt.style.use(args.mpl_style)
    args.centrality_specs = dict((
      ("indegrees", (
        r"$\log_{10}$indegree" if mpl.rcParams["text.usetex"] \
          else "log-indegree",
        np.log10,
        "nnodes",
        True,
        "Indegree")),
      ("pagerank", (
        r"$\log_{10}$pagerank" if mpl.rcParams["text.usetex"] \
          else "log-pagerank",
        np.log10,
        "nnodes_inv",
        True,
        "Pagerank")),
      ("harmonicc", (
        "harmonic centrality",
        None,
        "nnodes",
        False,
        "Harmonic Centrality")),
      ("closenessc", (
        r"closeness centrality $\times10^7$" if mpl.rcParams["text.usetex"] \
          else "closeness-centrality * 1e7",
        lambda x: x * 1e7,
        None, #"nnodes",
        False,
        "Closeness Centrality")),
    ))
    args.norm_str = {
        None: "",
        "nnodes": r"$/ n_{nodes}$",
        "nnodes_inv": r"$\times n_{nodes}$",
        "narcs": r"$/ n_{arcs}$",
        "narcs_inv": r"$\times n_{arcs}$",
    }

    # Dataset preprocessing function
    def preprocessed_dataset(graph, centrality: str, drop_other: bool = True):
      df = graph.supergenre_dataframe(
          **{centrality: getattr(graph, centrality)()})
      k, k_fn, _, _, _ = args.centrality_specs[centrality]
      if drop_other:
        df.drop((i for i, g in enumerate(df.genre) if g == "other"),
                inplace=True)

      if k is None:
        k = centrality
      else:
        if k_fn is None:
          k_fn = lambda x: x
        df[k] = k_fn(df[centrality])

      median_order = df.groupby(
          by=["genre"])[k].median().sort_values().iloc[::-1].index
      return df, k, median_order

    args.preprocessed_dataset = preprocessed_dataset

    # Load BVGraph
    jwebgraph.start_jvm(jvm_path=args.jvm_path)
    importlib.import_module("featgraph.jwebgraph.utils")
    args.graph = jwebgraph.utils.BVGraph(args.graph_path)
    logger.info("%s", args.graph)

    args.ref_artists = [args.graph.artist(aid=a) for a in args.ref_artists]

    # Prepare figure saving
    def fig_path(filename: str) -> str:
      return os.path.join(args.dest_path, filename)

    args.fig_path = fig_path

    def save_fig(filename: str, clf: bool = True):
      logger.info("Saving plot: %s", fig_path(filename))
      plt.savefig(fname=fig_path(filename))
      if clf:
        plt.clf()

    args.save_fig = save_fig

    def must_write(filename: str, *args_, **kwargs_):
      return args.overwrite or pathutils.notisfile(fig_path(filename), *args_,
                                                   **kwargs_)

    args.must_write = must_write

    # Load JSON settings
    if args.abbrev is None:
      args.abbrev = {}
    else:
      with open(args.abbrev, encoding="utf-8") as f:
        args.abbrev = json.load(f)
    args.abbrev = EchoDict(args.abbrev)

    palette_fname = args.palette
    args.palette = collections.defaultdict(lambda: "#cccccc")
    if palette_fname is not None:
      with open(palette_fname, encoding="utf-8") as f:
        for k, v in json.load(f).items():
          args.palette[k] = v

    return args

  @staticmethod
  def perform_threshold_comparison(args, plot_name_fn):
    """Perform comparison
    Args:
      args: Parsed CLI arguments
      plot_name_fn: Function that maps centrality key to plot file name"""
    sgc_graph = jwebgraph.utils.BVGraph(args.sgc_path)
    sgc = importlib.import_module("featgraph.sgc")
    tc = sgc.ThresholdComparison(
        sgc.ThresholdComparison.sgc_graph(sgc_graph),
        sgc.ThresholdComparison.spotify_graph(args.graph),
        thresholds=tuple(range(0, 81, 1)),
    )

    # Skip if all images are found
    if not (args.overwrite or pathutils.notisfile(
        list(map(args.fig_path, map(plot_name_fn, tc.centralities.values()))),
        func=lambda x: all(map(os.path.exists, x)),
    )):
      return None, False

    if args.csv_path is None or not os.path.isfile(args.csv_path):
      # Sample SGC graph
      if args.overwrite or \
        pathutils.notisglob(sgc_graph.path("*"),
                            msg="Found: %.40s... Skipping"):
        logger.info("Sampling SGC graph")
        sgc_model = sgc.SGCModel()
        sgc_nxgraph = sgc_model(seed=args.sgc_seed)
        logger.info("Converting nxgraph to BVGraph")
        sgc.to_bv(sgc_nxgraph, args.sgc_path)

      tc.threshold_graphs(tqdm=args.tqdm, overwrite=args.overwrite)
      tc.compute_centralities(tqdm=args.tqdm, overwrite=args.overwrite)

    logger.info("Centralities summary dataframe. File: %s ", args.csv_path)
    df = tc.dataframe(*(() if args.csv_path is None else (args.csv_path,)),
                      tqdm=args.tqdm,
                      overwrite=args.overwrite)
    df["nnodes_inv"] = 1 / df["nnodes"]

    return df, True
