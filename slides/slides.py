"""Plot utilities. This module requires matplotlib"""
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import importlib
import sys
import os
from chromatictools import cli
from featgraph import plots, scriptutils, jwebgraph, sgc, pathutils, logger
from typing import Optional


def _translate_log_ticks(
    offset: float,
    ax: Optional = None,
    add_zero: bool = True,
):
  r"""Move xticks by a specified amount on a log scale axis. Please, add
  :data:`{"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"}`
  to your matplotlib rcparams before calling this function

  Args:
    offset (float): Offset amount between position and ticks (difference
      between the tick position and the tick label value)
    ax: The axis. If not specified, the current axis is affected
    add_zero (bool): Add tick on the zero"""
  if ax is None:
    ax = plt.gca()
  xl = ax.get_xlim()
  xt = 10**np.arange(np.log10(xl[-1] + 1))
  if add_zero:
    xt = np.array([0, *xt])
  plt.gca().set_xticks(xt + offset)
  plt.gca().set_xticklabels(
      fr"$10^{'{'}{np.log10(i):.0f}{'}'}$" if i > 0 else r"$0^{\vphantom{1}}$"
      for i in xt)
  plt.xlim(xl)


def _hdi(p,
         x: Optional = None,
         ci: float = 0.94,
         n: Optional[int] = None,
         cdf: bool = False):
  """Compute the HDI around the mode

  Args:
    p (array): Array of the probabilities of x
    x (array): Array of variable values (if unspecified,
      the integer range of the same length of :data:`p` is used)
    ci (float): The minimum density of the HDI
    n (int): If specified, upsample the probability function
      to this number of values
    cdf (True): If :data:`True` interpret :data:`p`
      as a cumulative probability function"""
  if ci < 0 or ci > 1:
    raise ValueError("Invalid value for density. It should be "
                     f"a float between 0 and 1. Got {ci}")
  if cdf:
    p = np.diff([0, *p])
  if x is None:
    x = np.arange(len(p))
  if n is None:
    n = len(x)
  else:
    x_ = x
    p_ = p
    x = np.linspace(x_[0], x_[-1], n, endpoint=True)
    p = np.interp(x, x_, p_, left=0, right=0)

  def dx(idx, arr=x):
    l = idx if idx == 0 else idx - 1
    r = idx if idx == len(arr) - 1 else idx + 1
    return (arr[r] - arr[l]) / (r - l)

  l = np.argmax(p)
  r = l + 1
  while True:
    d_curr = sum(p[i] * dx(i) for i in range(l, r))
    if d_curr >= ci or (l == 0 and r == n):
      break
    if l == 0:
      r += 1
    elif r == n:
      l -= 1
    elif p[l - 1] >= p[r + 1]:
      l -= 1
    else:
      r += 1
  return x[l], x[r - 1]


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Make the plots for the slides"""
  parser = scriptutils.FeatgraphArgParse(
      description="Make plots for the presentation slides")
  parser.add_argument("graph_path", help="The base path for the BVGraph files")
  parser.add_argument(
      "dest_path", help="The destination path (directory) for the plot files")
  parser.add_argument("--csv-path",
                      default=None,
                      help="The path of the centralities summary dataframe")
  parser.add_argument("-F",
                      "--force",
                      dest="overwrite",
                      action="store_true",
                      help="Overwrite preexisting files")
  parser.add_argument("-a",
                      "--artist",
                      dest="ref_artists",
                      metavar="AID",
                      type=str,
                      action="append",
                      help="Specify a reference artist by ID")
  parser.add_argument("--sgc-path",
                      metavar="PATH",
                      default=".sgc-graph/sgc",
                      help="The path for the SGC random graph")
  parser.add_argument("--max-height",
                      metavar="H",
                      default=10.0,
                      type=float,
                      help="The maximum height of figures (in inches)")
  parser.add_argument("--max-width",
                      metavar="W",
                      default=10.0,
                      type=float,
                      help="The maximum width of figures (in inches)")
  parser.add_argument("--mpl-style",
                      metavar="FILEPATH",
                      default=os.path.join(os.path.dirname(__file__),
                                           "slides.mplstyle"),
                      help="The path of the matplotlib style file")
  args = parser.custom_parse(argv)

  def none_or_expand(s: Optional[str]) -> Optional[str]:
    return None if s is None else os.path.expanduser(s)

  args.jvm_path = none_or_expand(args.jvm_path)
  args.dest_path = none_or_expand(args.dest_path)
  args.graph_path = none_or_expand(args.graph_path)

  # Load BVGraph
  jwebgraph.start_jvm(jvm_path=args.jvm_path)
  importlib.import_module("featgraph.jwebgraph.utils")
  graph = jwebgraph.utils.BVGraph(args.graph_path)
  logger.info("%s", graph)

  # Prepare references for scatter plots
  ref_artists = [graph.artist(aid=a) for a in args.ref_artists]

  def scatter_refs(x, y, legend_kw: Optional[dict] = None, **kwargs):
    if len(ref_artists) == 0:
      return
    xs = x() if callable(x) else x
    xs = [xs[a.index] for a in ref_artists]
    ys = y() if callable(y) else y
    ys = [ys[a.index] for a in ref_artists]
    for xi, yi, ai in zip(xs, ys, ref_artists):
      plt.scatter(xi, yi, label=ai.name, **kwargs)
    plt.legend(**({} if legend_kw is None else legend_kw))

  # Prepare figure saving
  def figpath(filename: str) -> str:
    return os.path.join(args.dest_path, filename)

  def savefig(filename: str, clf: bool = True):
    logger.info("Saving plot: %s", figpath(filename))
    plt.savefig(fname=figpath(filename))
    if clf:
      plt.clf()

  # Configure matplotlib
  logger.info("Setting style: %s", args.mpl_style)
  plt.style.use(args.mpl_style)

  # Plot degrees
  degscatter_fname = "degrees.png"
  if args.overwrite or pathutils.notisfile(figpath(degscatter_fname)):
    logger.info("Plotting degrees")
    graph.compute_degrees()

    out_degs = graph.outdegrees()
    for i in range(len(out_degs)):
      out_degs[i] += 1
    plots.scatter(
        out_degs,
        graph.indegrees,
        marker=".",
        c=mpl.rcParams["lines.markerfacecolor"],
        xscale="log",
        yscale="log",
        label=graph.basename,
        xlabel="out-degree",
        ylabel="in-degree",
    )
    scatter_refs(out_degs, graph.indegrees, legend_kw=dict(loc="upper left"))
    del out_degs
    plt.gca().set_aspect("equal")
    plt.minorticks_off()
    _translate_log_ticks(1)
    savefig(degscatter_fname)

  # Plot distance probability functions
  neigh_fname = "distances.svg"
  if args.overwrite or pathutils.notisfile(figpath(neigh_fname)):
    logger.info("Compute neighbourhood")
    graph.compute_transpose()
    graph.compute_neighborhood()

    logger.info("Plotting distance probability mass function")
    d = graph.distances()
    d /= sum(d)
    plt.plot(d)
    d_mean = np.dot(np.arange(len(d)), d)
    p_mean = np.interp(d_mean, np.arange(len(d)), d)

    d_hdi_ci = 0.94
    d_hdi = _hdi(d, n=1024, ci=d_hdi_ci)
    plt.plot(d_hdi, np.zeros(2), c=mpl.rcParams["lines.color"], linewidth=3)

    fsize = mpl.rcParams["font.size"] + 2
    plt.text(d_hdi[0],
             p_mean * 0.02,
             f"{d_hdi[0]:.2f}",
             verticalalignment="bottom",
             horizontalalignment="right",
             fontsize=fsize)
    plt.text(d_hdi[1],
             p_mean * 0.02,
             f"{d_hdi[1]:.2f}",
             verticalalignment="bottom",
             horizontalalignment="left",
             fontsize=fsize)
    percent_s = r"$\%$" if mpl.rcParams["text.usetex"] else "%"
    plt.text(np.mean(d_hdi),
             p_mean * 0.125,
             f"{d_hdi_ci * 100:.0f}{percent_s} HDI",
             verticalalignment="center",
             horizontalalignment="center",
             fontsize=fsize)

    plt.text(d_mean,
             p_mean * 0.90,
             f"mean = {d_mean:.2f}",
             verticalalignment="center",
             horizontalalignment="center",
             fontsize=fsize)

    plt.xlabel("distance")
    plt.ylabel("probability")
    plt.title(f"{graph.basename}\nHyperBall ($log_2m$ = 8)")
    savefig(neigh_fname)

  # Compute indegrees dataframe
  indegrees_fname = "indegrees.svg"
  if args.overwrite or pathutils.notisfile(figpath(indegrees_fname)):
    logger.info("Load indegrees dataset")
    df = graph.supergenre_dataframe(indegree=graph.indegrees())
    df.drop((i for i, g in enumerate(df.genre) if g == "other"), inplace=True)
    k = r"$\log_{10}$indegree" if mpl.rcParams["text.usetex"] \
                               else "log-indegree"
    df[k] = np.log10(df["indegree"])
    violin_order = df.groupby(
        by=["genre"])[k].median().sort_values().iloc[::-1].index

    logger.info("Plot indegrees dataset")
    plt.xticks(rotation=33)
    sns.violinplot(data=df, x="genre", y=k, order=violin_order, cut=0)
    plt.gcf().set_size_inches(mpl.rcParams["figure.figsize"][1] *
                              np.array([16 / 9, 1]))
    savefig(indegrees_fname)

  # Compute transitions dataframe
  ## Sample SGC Graph
  seed = 42
  sgc_model = sgc.SGCModel()
  sgc_graph = jwebgraph.utils.BVGraph(args.sgc_path)
  if pathutils.notisglob(sgc_graph.path("*"), msg="Found: %.40s... Skipping"):
    logger.info("Sampling SGC graph")
    sgc_nxgraph = sgc_model(seed=seed)
    logger.info("Converting nxgraph to BVGraph")
    sgc.to_bv(sgc_nxgraph, args.sgc_path)

  ## Perform thresholing
  transisions_fnames_fmt = "transition-{}.svg".format
  tc = sgc.ThresholdComparison(
      sgc.ThresholdComparison.sgc_graph(sgc_graph),
      sgc.ThresholdComparison.spotify_graph(graph),
  )
  trasition_plot_kwargs = {
      "Harmonic Centrality": dict(norm="nnodes",),
      "Closeness Centrality": dict(norm="nnodes", logy=True),
      "Indegree": dict(norm="nnodes", logy=True),
      "Pagerank": dict(
          norm="nnodes_inv",
          logy=True,
      )
  }
  if args.overwrite or pathutils.notisfile(
      list(map(figpath, map(transisions_fnames_fmt, tc.centralities.values()))),
      func=lambda x: all(map(os.path.exists, x)),
  ):
    logger.info("Thresholding based on %s at thresholds: %s ", tc.attribute,
                tc.thresholds)
    tc.threshold_graphs(tqdm=args.tqdm, overwrite=args.overwrite)

    ## Compute centralities
    tc.compute_centralities(tqdm=args.tqdm, overwrite=args.overwrite)

    ## Build dataframe
    logger.info("Centralities summary dataframe. File: %s ", args.csv_path)
    df = tc.dataframe(*(() if args.csv_path is None else (args.csv_path,)),
                      tqdm=args.tqdm,
                      overwrite=args.overwrite)
    df["nnodes_inv"] = 1 / df["nnodes"]

    for k, v in tc.centralities.items():
      logger.info("Plotting %s", k)
      fig, ax = plt.subplots(1, 2, sharex=True)
      fig.set_size_inches(mpl.rcParams["figure.figsize"][1] *
                          np.array([20 / 9, 1]))
      plots.plot_centrality_transitions(df,
                                        k,
                                        cmap={
                                            "celebrities": "C0",
                                            "community leaders": "C1",
                                            "masses": "C2",
                                            "hip-hop": "C0",
                                            "classical": "C1",
                                            "rock": "C2",
                                        },
                                        fig=fig,
                                        ax=ax,
                                        **trasition_plot_kwargs.get(k, {}))
      savefig(transisions_fnames_fmt(v))
