"""Plot utilities. This module requires matplotlib"""
from matplotlib import pyplot as plt, patches
import matplotlib as mpl
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import importlib
import arviz
import sys
import os
from chromatictools import cli
from featgraph.misc import VectorOrCallable
from featgraph import bayesian_comparison, scriptutils, jwebgraph, logger
from typing import Optional, Callable, Dict, Tuple, Any, Sequence


def scatter(
    x: VectorOrCallable,
    y: VectorOrCallable,
    kendall_tau: bool = True,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xscale: str = "linear",
    yscale: str = "linear",
    label: Optional[str] = None,
    **kwargs,
) -> plt.Axes:
  r"""Scatterplot of two scores

  Args:
    x: The first vector or a function that returns the first vector
    y: The second vector or a function that returns the second vector
    kendall_tau (bool): If :data:`True` (default), then compute Kendall's
      :math:`\tau`
    ax (Axes): Plot axes
    xlabel (str): Label for the x axis
    ylabel (str): Label for the y axis
    xscale (str): Scaling for the x axis
    yscale (str): Scaling for the y axis
    label (str): Label for the graph
    kwargs: Keyword arguments for :func:`plt.scatter`

  Returns:
    Axes: Plot axes"""
  if ax is None:
    ax = plt.gca()
  ax.scatter(x() if callable(x) else x, y() if callable(y) else y, **kwargs)
  ax.set_xscale(xscale)
  ax.set_yscale(yscale)
  # Kendall Tau
  if kendall_tau:
    kt = importlib.import_module("featgraph.jwebgraph.utils").kendall_tau(x, y)
  else:
    kt = None
  # Make Title
  tit_li = []
  if label is not None:
    tit_li.append(label)
  if ylabel is not None:
    ax.set_ylabel(ylabel)
    tit_li.append(ylabel)
  if xlabel is not None:
    ax.set_xlabel(xlabel)
    if ylabel is not None:
      tit_li.append("vs")
    tit_li.append(xlabel)
  if len(tit_li) > 0:
    tit_li = [" ".join(tit_li)]
  if kt is not None:
    tit_li.append(fr"(Kendall $\tau$ = {kt:.5f})")
  if len(tit_li) > 0:
    ax.set_title("\n".join(tit_li))
  return ax


def draw_sgc_graph(
    g: nx.Graph,
    ax: Optional[plt.Axes] = None,
    masses_c="C0",
    celeb_c="C1",
    leader_c="C2",
    default_c="k",
    pos_fn: Callable[[nx.Graph], Dict[int, Tuple[float,
                                                 float]]] = nx.spring_layout,
    node_alpha: float = 1,
    edge_alpha: float = .1,
    draw_nodes: bool = False,
    legend: bool = True,
    seed: Optional[int] = None,
    edges_kwargs: Optional[Dict[str, Any]] = None,
    nodes_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
  """Draw a graph output with a Social Group Centrality model

  Args:
    g (Graph): The graph
    ax (Axes): The pyplot Axes onto which to draw the graph. Default is
      :data:`None` (use the output of :func:`plt.gca`)
    masses_c (color specification): Color for the *masses*
    celeb_c (color specification): Color for the *celebrities*
    leader_c (color specification): Color for the *community leaders*
    default_c (color specification): Color for other edges (should not happen)
    pos_fn (callable): Node positions function.
      Defaults to :func:`spring_layout`
    node_alpha (float): Opacity of nodes
    edge_alpha (float): Opacity of edges
    draw_nodes (bool): Whether to draw nodes or not
    legend (bool): Whether to add a legend to the plot or not
    seed (int): Seed for random number generator
    edges_kwargs (dict): Keyword arguments for :func:`draw_networkx_edges`
    nodes_kwargs (dict): Keyword arguments for :func:`draw_networkx_nodes`
    kwargs: Keyword arguments for draw functions

  Returns:
    Axes: The pyplot Axes onto which the graph is drawn"""
  if ax is None:
    ax = plt.gca()
  try:
    pos = pos_fn(g, seed=seed)
  except TypeError:
    pos = pos_fn(g)
  # decide edge colors
  edge_colors = {
      "masses": masses_c,
      "celebrities": celeb_c,
      "community leaders": leader_c,
      "other": default_c,
  }
  edge_lists = {k: [] for k in edge_colors}
  node_class = nx.get_node_attributes(g, "class")
  for e in g.edges():
    if all(node_class[v] == "masses" for v in e):
      edge_lists["masses"].append(e)
    elif not any(node_class[v] == "celebrities" for v in e):
      edge_lists["community leaders"].append(e)
    elif not any(node_class[v] == "community leaders" for v in e):
      edge_lists["celebrities"].append(e)
    else:
      edge_lists["other"].append(e)
  # draw edges
  for k, v in edge_lists.items():
    if v:
      c = edge_colors[k]
      nx.draw_networkx_edges(g,
                             pos,
                             ax=ax,
                             edgelist=v,
                             edge_color=c,
                             alpha=edge_alpha,
                             **({} if edges_kwargs is None else edges_kwargs),
                             **kwargs)
  # draw nodes
  if draw_nodes:
    for k, c in edge_colors.items():
      nodelist = [i for i, ki in g.nodes(data="class") if ki == k]
      if nodelist:
        nx.draw_networkx_nodes(g,
                               pos,
                               ax=ax,
                               nodelist=nodelist,
                               node_color=c,
                               alpha=node_alpha,
                               **({} if nodes_kwargs is None else nodes_kwargs),
                               **kwargs)
  if legend:
    ax.legend(handles=[
        patches.Patch(label=k, facecolor=edge_colors[k])
        for k in ("masses", "celebrities", "community leaders")
    ])
  return ax


def rope_matrix_plot(df: pd.DataFrame,
                     names: Optional[Sequence[str]] = None,
                     order=None,
                     x_label: str = "x",
                     y_label: str = "y",
                     probs_labels: Tuple[str, str, str] = (
                         "x - y < ROPE",
                         "x - y in ROPE",
                         "x - y > ROPE",
                     ),
                     th: Optional[float] = None,
                     normalize: bool = True,
                     legend: bool = False,
                     legend_selectors: Optional[Sequence[int]] = None,
                     ticks: bool = True):
  """Plot the ROPE probabilities as a RGB matrix

  Args:
    df (pd.DataFrame): ROPE probabilities dataframe
    names (sequence of str): Names of the different populations.
      If :data:`None`, the names are inferred from the dataframe
    order (callable): Key function for sorting populations. Alternatively,
      if :data:`"auto"`, the populations are sorted by decreasing difference
      of average probabilities above and below the ROPE
    x_label (str): Column name for first population names.
      Default is :data:`"x"`
    y_label (str): Column name for second population names.
      Default is :data:`"y"`
    probs_labels (triplet of str): Column names for the probabilities below,
      within and above the ROPE
    th (float): If specified, threshold hypoteses at the given value
    normalize (bool): If :data:`True` (default), normalize the RGB values.
      This results in brighter colors and better intelligibility
    legend (bool): If :data:`True`, plot a legend on the axis
    legend_selectors (sequence of int): Specifies which legend entries to keep
    ticks (bool): If :data:`True` (default), draw ticks and tick
      labels on both the x and y axis"""
  names = bayesian_comparison._rope_probabilities_names(  # pylint: disable=W0212
      df=df,
      names=names,
      order=order,
      x_label=x_label,
      y_label=y_label,
      probs_labels=probs_labels,
  )

  # Build matrix
  n = len(names)
  m = np.empty((n, n, 3))
  for i, j in itertools.product(range(n), repeat=2):
    if i == j:
      m[i, j, :] = (0, 1, 0)
      continue
    x = names[i]
    y = names[j]

    r = df
    r = r[r[x_label] == x]
    r = r[r[y_label] == y]
    for k, l in enumerate(probs_labels):
      m[i, j, k] = r[l].mean()
  if th is None:
    if normalize:
      m /= np.tile(np.expand_dims(np.max(m, axis=2), 2), (1, 1, 3))
  else:
    np.greater_equal(m, th, out=m)

  rv = plt.imshow(m)
  if ticks:
    plt.xticks(np.arange(n), names, rotation=90)
    plt.yticks(np.arange(n), names)

  if legend:
    # Build custom legend
    legend_patches = []
    legend_labels = []
    cols = filter(any, itertools.product(range(2), repeat=3))
    if th is None and not normalize:
      cols = (np.array(c) / sum(c) for c in cols)
    cols = list(cols)
    for c in cols:
      legend_patches.append(patches.Patch(facecolor=c))
      cs = "/".join(itertools.compress((r"$<$", r"$\in$", r"$>$"), c))
      legend_labels.append(f"row - column {cs} ROPE")

    if legend_selectors is None:
      # Choose which entries to show
      legend_selectors = range(len(cols))
      if th is not None:
        legend_selectors = filter(
            lambda ci: any(
                np.array_equal(cols[ci], m[i, j, :])
                for i, j in itertools.product(range(n), repeat=2)),
            legend_selectors)
      legend_selectors = list(legend_selectors)

    legend_patches = list(map(legend_patches.__getitem__, legend_selectors))
    legend_labels = list(map(legend_labels.__getitem__, legend_selectors))
    plt.legend(legend_patches, legend_labels)
  return rv


def plot_posterior(data,
                   names: Tuple[str, str],
                   es_rope: Tuple[float, float] = (-0.1, 0.1)):
  """Plot the posterior distribution of the pair
  comparison from the inference data

  Args:
    data: Inference data
    names (couple of str): The names of the two populations to compare
    es_rope (copule of float): The boundaries of the ROPE
      on the effect size. Default is :data:`(-0.1, 0.1)`"""
  arviz.plot_posterior(
      data,
      rope={f"effect size {names[0]} - {names[1]}": [{
          "rope": es_rope
      }]},
      ref_val={
          f"{k} {names[0]} - {names[1]}": [{
              "ref_val": 0
          }] for k in ("mean", "std")
      },
      var_names=[
          *[f"{g} {k}" for k in ("mean", "std") for g in names], "dof - 1", *[
              f"{k} {names[0]} - {names[1]}"
              for k in ("mean", "std", "effect size")
          ]
      ],
      grid=(2, 4),
  )


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
    cdf (True): If :data:`True` interpret :data:`p` as a cumulative probability function"""
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

  def dx(_i, _x=x):
    _l = _i if _i == 0 else _i - 1
    _r = _i if _i == len(_x) - 1 else _i + 1
    return (_x[_r] - _x[_l]) / (_r - _l)

  l = np.argmax(p)
  r = l + 1
  while True:
    d_curr = sum(p[i] * dx(i) for i in range(l, r))
    logger.info("[%d, %d] -> (%f, %f) = %f", l, r - 1, x[l], x[r - 1], d_curr)
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
  parser.add_argument("-a",
                      "--artist",
                      dest="ref_artists",
                      metavar="AID",
                      type=str,
                      action="append",
                      help="Specify a reference artist by ID")
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
  def savefig(filename: str, clf: bool = True):
    figpath = os.path.join(args.dest_path, filename)
    logger.info("Saving plot: %s", figpath)
    plt.savefig(fname=figpath)
    if clf:
      plt.clf()

  # Configure matplotlib
  logger.info("Setting style: %s", args.mpl_style)
  plt.style.use(args.mpl_style)

  # Plot degrees
  logger.info("Plotting degrees")

  out_degs = graph.outdegrees()
  for i in range(len(out_degs)):
    out_degs[i] += 1
  scatter(
      out_degs,
      graph.indegrees,
      marker=".",
      c="k",
      alpha=2**(-6),
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
  savefig("degrees.png")

  # Plot distance probability functions
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
  savefig("distances.png")
