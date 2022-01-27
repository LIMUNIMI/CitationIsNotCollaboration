"""Plot utilities"""
from matplotlib import pyplot as plt, patches
import matplotlib as mpl
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import importlib
import arviz
from featgraph.misc import VectorOrCallable
from featgraph import bayesian_comparison, metadata
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


def plot_centrality_transitions(
    df: pd.DataFrame,
    centrality_name: str,
    cmap: Optional[Dict[str, Any]] = None,
    centrality_name_key: str = "centrality",
    graph_names: Optional[Sequence[str]] = None,
    graph_name_key: str = "graph",
    type_name_key: str = "type_value",
    threshold_key: str = "threshold",
    threshold_attr: str = "popularity",
    norm=None,
    logy: bool = False,
    save: bool = False,
    aspect: float = 16 / 9,
    width: float = 6,
    figext: str = "svg",
    median: bool = True,
    mean_key: str = "mean",
    std_key: str = "std",
    quartile1_key: str = "quartile-1",
    median_key: str = "median",
    quartile3_key: str = "quartile-3",
    std_scale: float = 0.6744897501960817,  # 50% density
    fill_alpha: float = 0.25,
    fig=None,
    ax=None,
    legend_kw: Optional[dict] = None,
):
  """Plot centrality transitions for multiple graphs

  Args:
    df (DataFrame): Centralities summary
    centrality_name (str): Name of the centrality to plot
    cmap (dict): Map from type names to plot colors
    centrality_name_key (str): Name of the column in the dataframe
      with the centrality names
    graph_names (sequence of str): Names of the graphs, the centralities of
      which to plot. If unspecified, plot centralities for all graphs
      in the dataframe
    graph_name_key (str): Name of the column in the dataframe
      with the graph names
    type_name_key (str): Name of the column in the dataframe
      with the type names
    threshold_key (str): Name of the column in the dataframe
      with the threshold values
    threshold_attr (str): Name of the attribute used for thresholding
    norm (str): If specified, divide the centrality column
      by the column with this name. Use this for normalization
    logy (bool): If :data:`True`, set the y-axis as logarithmic
    aspect (float): Aspect ratio of the figure for a 1-graph plot
    width (float): Width of the figure
    median (bool): If :data:`True` (default), plot median and interquartile
      ranges. Otherwise, plot the mean and the range within :data:`std_scale`
      times the standard deviation
    mean_key (str): Name of the column in the dataframe
      with the mean values
    std_key (str): Name of the column in the dataframe
      with the standard deviation values
    quartile1_key (str): Name of the column in the dataframe
      with the quartile 1 values
    median_key (str): Name of the column in the dataframe
      with the median (quartile 2) values
    quartile3_key (str): Name of the column in the dataframe
      with the quartile 3 values
    std_scale (float): Scale factor for the standard deviation range
    fill_alpha (float): Fill opacity for the interquartile range or the
      standard deviation range
    fig: Figure onto which to plot
    ax: Array of axes onto which to plot
    save (bool): If :data:`True`, save the figure to a file. If a string,
      save to the specific filepath
    figext (str): Extension of the figure file
    legend_kw (dict): Keyword arguments for legend"""
  if graph_names is None:
    graph_names = df[graph_name_key].unique()
  if fig is None:
    fig = plt.gcf()
    fig.set_size_inches(width * np.array([aspect, len(graph_names)]))
  if ax is None:
    ax = fig.subplots(nrows=len(graph_names), sharex=True)
  if cmap is None:
    cmap = {}
    for graph_name in graph_names:
      for tv, c in zip(
          df[df[graph_name_key] == graph_name][type_name_key].unique(),
          itertools.cycle(mpl.rcParams["axes.prop_cycle"])):
        cmap[tv] = c["color"]

  for a, graph_name in zip(ax, graph_names):
    if logy:
      a.set_yscale("log")

    def data_iterator(df=df[(df[graph_name_key] == graph_name) &
                            (df[centrality_name_key] == centrality_name)]):
      for k in df[type_name_key].unique():
        dfk = df[df[type_name_key] == k]
        thresholds = dfk[threshold_key].to_numpy()
        idx = np.argsort(thresholds)
        thresholds = thresholds[idx]

        if median:
          kq1 = dfk[quartile1_key].to_numpy()[idx]
          kq2 = dfk[median_key].to_numpy()[idx]
          kq3 = dfk[quartile3_key].to_numpy()[idx]
        else:
          kq2 = dfk[mean_key].to_numpy()[idx]
          ks = dfk[std_key].to_numpy()[idx] * std_scale
          kq1 = kq2 - ks
          kq3 = kq2 + ks
        if norm:
          kn = dfk[norm].to_numpy()[idx]
          kq1 /= kn
          kq2 /= kn
          kq3 /= kn
        yield k, thresholds, kq1, kq2, kq3

    for k, thresholds, _, kq2, _ in data_iterator():
      if logy:
        kq2[np.argwhere(np.less_equal(kq2, 0))] = np.nan
      a.plot(thresholds, kq2, label=k, c=cmap.get(k, "k"))
    yl = a.get_ylim()
    for k, thresholds, kq1, _, kq3 in data_iterator():
      a.fill_between(
          thresholds,
          np.clip(kq1, *yl),
          np.clip(kq3, *yl),
          facecolor=cmap.get(k, "k"),
          alpha=fill_alpha,
      )
    a.set_ylim(*yl)
    a.legend(**({} if legend_kw is None else legend_kw))
    a.set_title(graph_name)
    a.set_ylabel(centrality_name)
    a.set_xlabel(f"{threshold_attr} threshold")

  if save:
    fpath = f"compare-{centrality_name}" + \
          (f"-norm_{norm}" if norm else "") + \
          ("-median" if logy else "") + \
          ("-semilogy" if logy else "") + \
          f".{figext}" if isinstance(save, bool) else save
    plt.savefig(fpath, bbox_inches="tight")


def translate_log_ticks(
    offset: float,
    ax: Optional[plt.Axes] = None,
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


def hdi_monomodal(p,
                  x: Optional[Sequence] = None,
                  d: float = 0.94,
                  n: Optional[int] = None,
                  cdf: bool = False):
  """Compute the HDI around the mode

  Args:
    p (array): Array of the probabilities of x
    x (array): Array of variable values (if unspecified,
      the integer range of the same length of :data:`p` is used)
    d (float): The minimum density of the HDI
    n (int): If specified, upsample the probability function
      to this number of values
    cdf (True): If :data:`True` interpret :data:`p`
      as a cumulative probability function"""
  if d < 0 or d > 1:
    raise ValueError("Invalid value for density. It should be "
                     f"a float between 0 and 1. Got {d}")
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
    if d_curr >= d or (l == 0 and r == n):
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


def scatter_refs(x: VectorOrCallable,
                 y: VectorOrCallable,
                 ref_artists: Sequence[metadata.Artist],
                 legend_kw: Optional[dict] = None,
                 **kwargs):
  """Scatter plot values for reference artists

  Args:
    x (array or callable): x-coordinate values for all nodes
    y (array or callable): y-coordinate values for all nodes
    ref_artist (sequence of Artist): artists to include as references
    legend_kw (dict): Keyword arguments for the legend
    kwargs: Keyword arguments for the scatter plot"""
  if len(ref_artists) == 0:
    return
  xs = x() if callable(x) else x
  xs = [xs[a.index] for a in ref_artists]
  ys = y() if callable(y) else y
  ys = [ys[a.index] for a in ref_artists]
  for xi, yi, ai in zip(xs, ys, ref_artists):
    plt.scatter(xi, yi, label=ai.name, **kwargs)
  plt.legend(**({} if legend_kw is None else legend_kw))


def degrees_scatterplot(graph: "featgraph.jwebgraph.utils.BVGraph",
                        ref_artists: Optional[Sequence[metadata.Artist]] = None,
                        ref_kwargs: Optional[dict] = None,
                        **kwargs):
  """Scatterplot of indegree vs outdegree

  Args:
    graph (BVGraph): Graph whose degrees to plot
    ref_artists (sequence of Artist): Reference artists
    ref_kwargs (dict): Keyword arguments for reference artists scatterplot
    kwargs: Keyword arguments for scatterplot"""
  outdegrees = graph.outdegrees()
  # Outdegree can be 0 and it wouldn't be plotted
  # So, make it 1 and modify axis labels
  for i in range(len(outdegrees)):
    outdegrees[i] += 1
  scatter(outdegrees,
          graph.indegrees,
          marker=".",
          c=mpl.rcParams["lines.markerfacecolor"],
          xscale="log",
          yscale="log",
          label=graph.basename,
          xlabel="out-degree",
          ylabel="in-degree",
          **kwargs)
  if ref_artists is not None:
    if ref_kwargs is None:
      ref_kwargs = {}
    scatter_refs(outdegrees, graph.indegrees, ref_artists, **ref_kwargs,
                 **kwargs)
  del outdegrees
  plt.gca().set_aspect("equal")
  plt.minorticks_off()
  translate_log_ticks(1)
  bs = "\\"
  rec = f"Reciprocity ${bs}rho = ${graph.reciprocity():.5f}"
  plt.gca().set_title(f"{plt.gca().get_title()[:-1]}, {rec})")


def _dict_copy_union(d: Optional[dict] = None, **kwargs):
  """Return a copy of a dictionary, eventually adding key-value pairs

  Args:
    d (dict): A dicitonary to copy or :data:`None` (start with
      an empty dictionary)
    kwargs: Key-value pairs to add to :data:`d` only if the
      keys are not already in d

  Returns:
    dict: The copy dicitonary"""
  d = {} if d is None else dict(d.items())
  for k, v in kwargs.items():
    if k not in d:
      d[k] = v
  return d


def degrees_jointplot(graph: "featgraph.jwebgraph.utils.BVGraph",
                      log_p1: bool = True,
                      log_marginal: bool = False,
                      xlabel: Optional[str] = "Outdegree",
                      ylabel: Optional[str] = "Indegree",
                      ref_artists: Optional[Sequence[metadata.Artist]] = None,
                      scatter_kws: Optional[dict] = None,
                      text_kws: Optional[dict] = None,
                      marginal_kws: Optional[dict] = None,
                      stats_kw: Optional[dict] = None,
                      zorder: int = 100,
                      grid: Optional = None,
                      kendall_tau: bool = False,
                      reciprocity: bool = False,
                      **kwargs):
  """Jointplot of indegree vs outdegree

  Args:
    graph (BVGraph): Graph whose degrees to plot
    log_p1 (bool): If :data:`True` (default), then plot the values scaled as
      :data:`log10(x + 1)`
    log_marginal (bool): If :data:`True` (default), then plot the marginal
      frequencies on a logarithmic scale
    xlabel (str): Label for the x (out-degree) axis
    ylabel (str): Label for the y (in-degree) axis
    ref_artists (sequence of Artist): Reference artists
    scatter_kwargs (dict): Keyword arguments for reference artists scatterplot
    text_kwargs (dict): Keyword arguments for reference artists text
    marginal_kws (dict):  Keyword arguments for marginal distributions plot
    stats_kws (dict):  Keyword arguments for statistics text (additional
      w.r.t. :data:`text_kwargs`)
    zorder (int): Base zorder for plots
    grid: Single argument or keyword arguments for turning on grids
    kendall_tau (bool): If :data:`True`, add a text with the value of the
      Kendall Tau correlation coefficient
    reciprocity (bool): If :data:`True`, add a text with the value of the
      reciprocity
    kwargs: Keyword arguments for jointplot"""
  # Load data
  df = pd.DataFrame(data={
      "outdegree": graph.outdegrees(),
      "indegree": graph.indegrees(),
  })
  if log_p1:
    kx = "log(outdegree+1)"
    ky = "log(indegree+1)"
    df[kx] = np.log10(df["outdegree"] + 1)
    df[ky] = np.log10(df["indegree"] + 1)
  else:
    kx = "outdegree"
    ky = "indegree"

  # Plot
  kwargs["zorder"] = zorder
  marginal_kws = _dict_copy_union(marginal_kws, zorder=zorder)
  if "bins" in kwargs:
    marginal_kws["bins"] = marginal_kws.get("bins", kwargs["bins"])
  jp = sns.jointplot(data=df, x=kx, y=ky, marginal_kws=marginal_kws, **kwargs)

  # Scatter refartists
  if ref_artists is not None:
    ref_rows = df.iloc[[a.index for a in ref_artists]]
    # Scatter
    scatter_kws = _dict_copy_union(scatter_kws, zorder=zorder + 1)
    sns.scatterplot(ax=jp.ax_joint, data=ref_rows, x=kx, y=ky, **scatter_kws)
    # Text
    jp_cx = np.mean(jp.ax_joint.get_xlim())
    jp_cy = np.mean(jp.ax_joint.get_ylim())
    for a, (_, a_row) in zip(ref_artists, ref_rows.iterrows()):
      a_x = a_row[kx]
      a_y = a_row[ky]
      a_text_kws = _dict_copy_union(
          text_kws,
          zorder=zorder + 2,
          horizontalalignment="left" if a_x < jp_cx else "right",
          verticalalignment="bottom" if a_y < jp_cy else "top")
      jp.ax_joint.text(a_x, a_y, a.name, **a_text_kws)

  # Write stats
  stats_l = []
  if kendall_tau:
    kt = importlib.import_module("featgraph.jwebgraph.utils").kendall_tau(
        graph.outdegrees, graph.indegrees)
    stats_l.append(r"Kendall $\tau = " + f"{kt:.3f}" + r"$")
  if reciprocity:
    r = graph.reciprocity()
    stats_l.append(r"Reciprocity $\rho = " + f"{r:.3f}" + r"$")
  if len(stats_l) > 0:
    stats_s = "\n".join(stats_l)
    stats_kw = _dict_copy_union(stats_kw,
                                x=0.975,
                                y=0.975,
                                s=stats_s,
                                horizontalalignment="right",
                                verticalalignment="top")
    stats_kw = _dict_copy_union(stats_kw, **_dict_copy_union(text_kws))
    jp.fig.text(**stats_kw)

  # Log-scale histograms
  if log_marginal:
    jp.ax_marg_x.set_yscale("log")
    jp.ax_marg_y.set_xscale("log")

  # Adjust labels
  jp.set_axis_labels(xlabel=xlabel, ylabel=ylabel)
  jp.ax_joint.xaxis.set_label_coords(0.5, 1.0025)
  jp.ax_joint.xaxis.label.set_verticalalignment("bottom")
  jp.ax_joint.yaxis.set_label_coords(1.0025, 0.5)
  jp.ax_joint.yaxis.label.set_rotation(-90)
  jp.ax_marg_x.tick_params(axis="y", reset=True)
  jp.ax_marg_y.tick_params(axis="x", reset=True)
  if log_p1:
    # Get ticks on powers of 10, their proportional middles, and on zero
    tick_values = np.floor(
        2 * np.max([*jp.ax_marg_x.get_xlim(), *jp.ax_marg_y.get_ylim()]))
    tick_values = np.log10(np.power(10, np.arange(tick_values + 1) / 2) + 1)
    tick_values = np.array([0, *tick_values])

    tick_labels = np.array([
        f"$10^{(i - 1) // 2:.0f}$" if i % 2 == 1 else ("" if i else "$0$")
        for i in range(len(tick_values))
    ])

    # Filter ticks outside limits
    def ticks_between(a, b):
      idx = [i for i, v in enumerate(tick_values) if a < v < b]
      return tick_values[idx], tick_labels[idx]

    jp.ax_joint.set_xticks(*ticks_between(*jp.ax_joint.get_xlim()))
    jp.ax_joint.set_yticks(*ticks_between(*jp.ax_joint.get_ylim()))

  # Turn on grids
  if grid is not None:
    for a in (jp.ax_joint, jp.ax_marg_x, jp.ax_marg_y):
      if hasattr(grid, "items"):
        a.grid(**grid)
      else:
        a.grid(grid)
  jp.fig.set_size_inches(mpl.rcParams["figure.figsize"])
  return jp


def plot_distances(graph: "featgraph.jwebgraph.utils.BVGraph",
                   probability: bool = True,
                   hdi: float = 0.94):
  """Plot the distribution of distances in the graph

  Args:
    graph (BVGraph): Graph whose distance distribution to plot
    probability (bool): If :data:`True` (default), then normalize sum to 1.
      Otherwise plot absolute counts
    hdi (float): Density of the High-Density Interval"""
  d = graph.distances()
  if probability:
    d /= sum(d)
  plt.plot(d)

  plt.xlabel("distance")
  if probability:
    plt.ylabel("probability")
  else:
    plt.ylabel("frequency")
    d /= sum(d)
  d_mean = np.dot(np.arange(len(d)), d)
  p_mean = np.interp(d_mean, np.arange(len(d)), d)

  fsize = mpl.rcParams["font.size"] + 2
  percent_s = r"$\%$" if mpl.rcParams["text.usetex"] else "%"

  if hdi:
    d_hdi = hdi_monomodal(d, n=max(1024, len(d)), d=hdi)
    plt.plot(d_hdi, np.zeros(2), c=mpl.rcParams["lines.color"], linewidth=3)
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
    plt.text(np.mean(d_hdi),
             p_mean * 0.125,
             f"{hdi * 100:.0f}{percent_s} HDI",
             verticalalignment="center",
             horizontalalignment="center",
             fontsize=fsize)
  plt.text(d_mean,
           p_mean * 0.90,
           f"mean = {d_mean:.2f}",
           verticalalignment="center",
           horizontalalignment="center",
           fontsize=fsize)


class ExponentFormatter(mpl.ticker.ScalarFormatter):
  """Ticks formatter for setting an explicit exponent

  Args:
    args: Positional arguments for :class:`ScalarFormatter`
    exponent (int): Explicit exponent
    kwargs: Keyword arguments for :class:`ScalarFormatter`"""

  def __init__(self, *args, exponent: Optional[int] = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.orderOfMagnitude = exponent

  def _set_order_of_magnitude(self):
    if self.orderOfMagnitude is None:
      super()._set_order_of_magnitude()
