"""Compare centrality values across genres"""
from featgraph import scriptutils, jwebgraph, logger
from chromatictools import cli
import importlib
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

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
  args = parser.custom_parse(argv)
  if args.metric not in metrics_dict:
    logger.error(
        "Undefined metric '%s'. Please, run with "
        "--help for a list of supported metrics", args.metric)
    return 1

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
  if args.violin_path:
    violin_order = df.groupby(
        by=["genre"])[k].median().sort_values().iloc[::-1].index
    plt.xticks(rotation=33)
    sns.violinplot(data=df, x="genre", y=k, order=violin_order)
    logger.info("Saving violin plot to '%s'", args.violin_path)
    size_inches = np.array([args.violin_aspect, 1.0])
    size_inches = size_inches * args.violin_scale / np.sqrt(
        np.sum(np.square(size_inches)))
    plt.gcf().set_size_inches(size_inches)
    plt.savefig(args.violin_path)
    plt.clf()
