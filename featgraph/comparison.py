"""Compare centrality values across genres"""
from featgraph import scriptutils, jwebgraph, bayesian_comparison, pathutils, logger
import logging
from chromatictools import cli
import itertools
import importlib
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import arviz

metrics_dict = {
    "closeness":
        (lambda g, _: g.closenessc(), lambda g, _: g.compute_closenessc()),
    "harmonic":
        (lambda g, _: g.harmonicc(), lambda g, _: g.compute_harmonicc()),
    "indegree": (lambda g, _: g.indegrees(), lambda g, _: g.compute_degrees()),
    "pagerank": (lambda g, a: g.pagerank(alpha=a.pagerank_alpha),
                 lambda g, a: g.compute_pagerank(alpha=a.pagerank_alpha)),
}


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Run comparison script"""
  parser = scriptutils.FeatgraphArgParse(
      description="Compare centrality values across "
      "musical genres with Bayesian modeling")
  parser.add_argument("graph_path", help="The base path for the BVGraph files")
  parser.add_argument(
      "metric",
      help="The name of the metric to compare. "
      f"Available metrics are: {', '.join(sorted(metrics_dict))}")
  parser.add_argument("--output-path",
                      "-O",
                      metavar="PATH",
                      default=None,
                      help="The path for saving inference data to file")
  parser.add_argument("--violin-path",
                      metavar="PATH",
                      default=None,
                      help="The path for saving the violin plot to file")
  parser.add_argument("--violin-aspect",
                      metavar="RATIO",
                      type=float,
                      default=16 / 9,
                      help="The aspect ratio for the violin plot")
  parser.add_argument("--violin-scale",
                      metavar="SCALE",
                      type=float,
                      default=18.0,
                      help="The scale for the violin plot (in inches)")
  parser.add_argument("--pagerank-alpha",
                      "-a",
                      type=float,
                      default=0.85,
                      metavar="A",
                      help="The alpha parameter for pagerank")
  parser.add_argument("--logarithm",
                      "-L",
                      action="store_true",
                      help="Compute the logarithm (base 10) of the metric")
  parser.add_argument("--keep-other",
                      action="store_true",
                      help="Keep the genre 'other'")
  parser.add_argument("--sample-chains",
                      type=int,
                      default=4,
                      help="Number of Markov chains for NUTS")
  parser.add_argument("--sample-cores",
                      type=int,
                      default=None,
                      help="Number of cores for NUTS. "
                      "Defaults to the number of Markov chains")
  parser.add_argument("--sample-draws",
                      type=int,
                      default=4000,
                      help="Number of draws from NUTS")
  parser.add_argument("--netcdf-path",
                      metavar="PATH",
                      default=None,
                      help="The directory path for saving the "
                      "inference data to file")
  parser.add_argument("--netcdf-compress",
                      action="store_true",
                      help="Compress netcdf file")
  args = parser.custom_parse(argv)
  if args.metric not in metrics_dict:
    logger.error(
        "Undefined metric '%s'. Please, run with "
        "--help for a list of supported metrics", args.metric)
    return 1
  if args.sample_cores is None:
    args.sample_cores = args.sample_chains
  tqdm_fn = (lambda x, *args, **kwargs: x) if args.tqdm is None else args.tqdm

  # Load BVGraph
  jwebgraph.start_jvm(jvm_path=args.jvm_path)
  importlib.import_module("featgraph.jwebgraph.utils")
  graph = jwebgraph.utils.BVGraph(args.graph_path)

  # Compute metrics and load in a dataframe
  get_fn, compute_fn = metrics_dict[args.metric]
  graph.compute_transpose()
  compute_fn(graph, args)
  df = graph.supergenre_dataframe(**{args.metric: get_fn(graph, args)})
  if not args.keep_other:
    df.drop((i for i, g in enumerate(df.genre) if g == "other"), inplace=True)
  if args.logarithm:
    k = f"log_{args.metric}"
    df[k] = np.log10(df[args.metric])
  else:
    k = args.metric
  logger.info("Dataframe:\n%s", df)
  violin_order = df.groupby(
      by=["genre"])[k].median().sort_values().iloc[::-1].index
  if args.violin_path:
    plt.xticks(rotation=33)
    sns.violinplot(data=df, x="genre", y=k, order=violin_order)
    logger.info("Saving violin plot to '%s'", args.violin_path)
    size_inches = np.array([args.violin_aspect, 1.0])
    size_inches = size_inches * args.violin_scale / np.sqrt(
        np.sum(np.square(size_inches)))
    plt.gcf().set_size_inches(size_inches)
    plt.savefig(args.violin_path)
    plt.clf()

  for m in ("filelock",):
    importlib.import_module(m)
    logging.getLogger(m).setLevel("WARN")
  # Perform comparison
  genres = sorted(df["genre"].unique(), key=violin_order.to_list().index)
  # tcomp_data = {}
  if args.netcdf_path is not None:
    os.makedirs(args.netcdf_path, exist_ok=True)
  for gx, gy in tqdm_fn(itertools.combinations(genres, 2),
                        total=len(genres) * (len(genres) - 1) // 2):
    # if gx not in tcomp_data:
    #   tcomp_data[gx] = {}
    filepath_kh = None if args.netcdf_path is None else os.path.join(
        args.netcdf_path, f"{gx}-{gy}.netcdf")
    if filepath_kh is None or pathutils.notisfile(filepath_kh):
      tcomp = bayesian_comparison.StudentTComparison().fit(
          **{g: df[df["genre"] == g][k] for g in (gx, gy)})
      # tcomp_data[gx][gy] =
      tcomp_data = tcomp.sample(chains=args.sample_chains,
                                        cores=args.sample_cores,
                                        draws=args.sample_draws,
                                        return_inferencedata=True,
                                        progressbar=args.tqdm is not None)
      if filepath_kh is not None:
        logger.info("Saving netcdf file: %s (compress=%s)", filepath_kh,
                    args.netcdf_compress)
        # tcomp_data[gx][gy]
        tcomp_data.to_netcdf(filepath_kh, compress=args.netcdf_compress)
    else:
      logger.info("Loading netcdf file: %s", filepath_kh)
      tcomp_data = arviz.from_netcdf(filepath_kh)
    # TODO: Do something with tcomp_data
