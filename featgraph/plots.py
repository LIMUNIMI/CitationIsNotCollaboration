"""Plot utilities. This module requires matplotlib"""
from matplotlib import pyplot as plt, patches
import matplotlib as mpl
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import importlib
import arviz
from featgraph.misc import VectorOrCallable
from featgraph import bayesian_comparison
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
    std_scale: float = 0.7,
    fill_alpha: float = 0.25,
    fig=None,
    ax=None,
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
    figext (str): Extension of the figure file"""
  if fig is None:
    fig = plt.gcf()
  if graph_names is None:
    graph_names = df[graph_name_key].unique()
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
    df_ = df[(df[graph_name_key] == graph_name) &
             (df[centrality_name_key] == centrality_name)]
    for k in df_[type_name_key].unique():
      dfk = df_[df_[type_name_key] == k]
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
      a.plot(thresholds, kq2, label=k, c=cmap.get(k, "k"))
      a.fill_between(
          thresholds,
          kq1,
          kq3,
          facecolor=cmap.get(k, "k"),
          alpha=fill_alpha,
      )
    if logy:
      a.set_yscale("log")
    a.legend()
    a.set_title(graph_name)
    a.set_ylabel(centrality_name)
    a.set_xlabel(f"{threshold_attr} threshold")

  fig.set_size_inches(width * np.array([aspect, len(graph_names)]))
  if save:
    fpath = f"compare-{centrality_name}" + \
          (f"-norm_{norm}" if norm else "") + \
          (f"-median" if logy else "") + \
          (f"-semilogy" if logy else "") + \
          f".{figext}" if isinstance(save, bool) else save
    plt.savefig(fpath, bbox_inches="tight")
