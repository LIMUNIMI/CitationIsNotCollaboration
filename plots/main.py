"""Make the plots for the paper or for the slides"""
import functools
import itertools
from featgraph import scriptutils, plots, logger
from chromatictools import cli
from matplotlib import pyplot as plt, colors
import matplotlib as mpl
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Make the plots for the paper"""
  cwd = os.path.dirname(__file__)
  parser = scriptutils.FeatgraphPlotsArgParse(
      style_file=os.path.join(cwd, "paper.mplstyle"),
      abbrev_file=os.path.join(cwd, "genre_abbreviations.json"),
      palette_file=os.path.join(cwd, "genre_palette.json"),
      description=__doc__)
  args = parser.custom_parse(argv)
  graph = args.graph

  # --- Violin plots by genre -------------------------------------------------
  violin_fname = "violinplots"
  if args.must_write(violin_fname):
    # Compute centralities
    graph.compute_transpose()
    graph.compute_symmetrized()

    centralities = (
        "indegrees",
        "pagerank",
        "harmonicc",
        "linc",
        "closenessc",
    )
    for c in args.centrality_specs:
      getattr(graph, f"compute_{c}")()
    logger.info("Loading centralities dataframe")
    df = graph.supergenre_dataframe(
        **{c: getattr(graph, c)() for c in centralities})

    # Preprocess dataframe
    logger.info("Preprocessing centralities dataframe")
    df.drop((i for i, g in enumerate(df.genre) if g == "other"), inplace=True)
    for k, (fn_k, fn, _, _, _) in args.centrality_specs.items():
      if k in df.columns:
        if fn is None:
          fn = lambda x: x
        df[fn_k] = fn(df[k])

    centralities_titles = tuple(
        args.centrality_specs[c][0] for c in centralities)

    logger.info("Plotting centralities violin plots")
    # Structure figure
    _, axs = plt.subplots(
        1,
        len(centralities),
        figsize=np.array([1, 10 / 16]) * mpl.rcParams["figure.figsize"][0] * 2,
        gridspec_kw=dict(wspace=0.32),
    )

    axs[centralities.index("harmonicc")].xaxis.set_major_formatter(
        plots.ExponentFormatter(exponent=5))
    axs[centralities.index("linc")].xaxis.set_major_formatter(
        plots.ExponentFormatter(exponent=5))

    # Plot
    for i, (ax, k) in enumerate(zip(axs, centralities_titles)):
      plots.violinplot_set(sns.violinplot(ax=ax,
                                          data=df,
                                          y="genre",
                                          x=k,
                                          order=plots.median_order(
                                              df, sort_by=k, group_by="genre"),
                                          cut=0,
                                          orient="h",
                                          palette=args.palette_desaturated),
                           zorder=100,
                           ec=functools.partial(colors.to_rgba, alpha=0.125))
      ax.xaxis.set_label_position("top")
      ax.grid(axis="x")
      yt_labels = ax.get_yticklabels()
      ax.set_ylabel("")
      if i:
        for ytl in yt_labels:
          ytl.set_text(args.abbrev[ytl.get_text()])
      ax.set_yticklabels(yt_labels, size="small")
    args.save_fig(violin_fname)
  # ---------------------------------------------------------------------------

  # --- Reciprocity violin plots by genre -------------------------------------
  r_violin_fname = "reciprocity-violinplots"
  if args.must_write(r_violin_fname):
    # Compute reciprocity
    graph.compute_transpose()
    graph.compute_degrees()
    graph.compute_reciprocity()

    logger.info("Loading reciprocity dataframe")
    k = "reciprocity"
    df = graph.supergenre_dataframe(**{k: graph.reciprocity()})
    df.drop(df.index[np.isnan(df[k])], inplace=True)
    df.reset_index(inplace=True)
    df.drop((i for i, g in enumerate(df.genre) if g == "other"), inplace=True)
    df.reset_index(inplace=True)

    # Plot
    logger.info("Plotting reciprocity violin plots")
    plots.violinplot_set(sns.violinplot(data=df,
                                        x="genre",
                                        y="reciprocity",
                                        order=plots.median_order(
                                            df, sort_by=k, group_by="genre"),
                                        cut=0,
                                        orient="v",
                                        palette=args.palette_desaturated),
                         zorder=100,
                         ec=functools.partial(colors.to_rgba, alpha=0.125))
    # plt.gca().yaxis.set_label_position("top")
    plt.gca().set_xlabel(None)
    plt.gca().set_ylabel(None)
    plt.gca().set_title("reciprocity")
    plt.gca().grid(axis="y")
    plt.gca().tick_params(labelsize="small")

    plt.gcf().set_size_inches(
        np.array([1, 0.25]) * mpl.rcParams["figure.figsize"][0] * 2)
    args.save_fig(r_violin_fname)
  # ---------------------------------------------------------------------------

  # --- Centrality transitions ------------------------------------------------
  transition_plot_fname = "centrality-transitions"
  cc_sizes_fname = "cc-transitions"
  df, plot_comparison = parser.perform_threshold_comparison(
      args, transition_plot_fname, cc_sizes_fname)
  if plot_comparison:
    for fname, centralities, scale, legend_kw, subplot_kw in (
        (
            transition_plot_fname,
            (
                "indegrees",
                "pagerank",
                "harmonicc",
                "linc",
                "closenessc",
            ),
            2,
            dict(loc="upper left"),
            dict(sharex=True),
        ),
        (
            cc_sizes_fname,
            (
                "node_scc_sizes",
                "node_wcc_sizes",
            ),
            1,
            dict(loc="lower left"),
            dict(sharex=True, sharey=True),
        ),
    ):
      if args.must_write(fname):
        n = len(centralities)
        fig, ax = plt.subplots(2, n, **subplot_kw)
        for i, k_s in enumerate(centralities):
          _, _, norm, logy, k = args.centrality_specs[k_s]
          logger.info("Plotting %s", k)
          plots.plot_centrality_transitions(
              df,
              k,
              median=False,
              norm=norm,
              logy=logy,
              fill_alpha=0.1,
              cmap={
                  "celebrities": args.palette["hip-hop"],
                  "community leaders": args.palette["classical"],
                  "masses": args.palette["rock"],
                  "hip-hop": args.palette["hip-hop"],
                  "classical": args.palette["classical"],
                  "rock": args.palette["rock"],
              },
              legend_kw=legend_kw,
              ax=ax[:, i])
          for a in ax[:, i]:
            yl = a.set_ylabel("" if i else a.get_title())
            yl.set_rotation(0)
            yl.set_horizontalalignment("left")
            a.yaxis.set_label_coords(-0.05, 1.05)
            a.set_title("")
            a.grid()
            if i == n - 1:
              pass
            else:
              a.get_legend().remove()
          ax[0, i].set_xlabel("")
          ax[0, i].set_title("\n".join(
              (k, args.norm_str.get(norm, f"normalized by {norm}"))))
        fig.set_size_inches(mpl.rcParams["figure.figsize"][0] * scale *
                            np.array([1, 2 / n]))
        args.save_fig(fname)
  # ---------------------------------------------------------------------------

  # --- Centrality correlations -----------------------------------------------
  cc_cc_plot_fname = "cc-correlations"
  df, plot_comparison = parser.perform_threshold_comparison(
      args, cc_cc_plot_fname)
  if plot_comparison:
    if args.must_write(cc_cc_plot_fname):
      logger.info("Plotting Centrality Correlations with SCC size")
      _, fig, axs = plots.centrality_correlations(
          pd.concat(tuple(plots.preprocessed_additions(df))),
          "Strongly Connected Component Size" + plots.div_suffix,
          [
              "Indegree" + plots.div_suffix,
              "Pagerank" + plots.mul_suffix,
              "Harmonic Centrality" + plots.div_suffix,
              "Lin Centrality" + plots.div_suffix,
              "Closeness Centrality" + plots.div_suffix,
          ],
          cc_fn=stats.kendalltau,
          p_thresholds=(.001,),
          plt_kws=dict(edgecolor="none", alpha=1 / 3, zorder=100),
          subplot_kws=dict(sharey="row"),
          palette={
              "celebrities": args.palette["hip-hop"],
              "community leaders": args.palette["classical"],
              "masses": args.palette["rock"],
              "hip-hop": args.palette["hip-hop"],
              "classical": args.palette["classical"],
              "rock": args.palette["rock"],
          },
      )
      for i, ax in enumerate(itertools.chain.from_iterable(axs)):
        ax.grid()
        if i % 5 == 4:
          sns.move_legend(ax, "upper right")
        else:
          sns.move_legend(ax, "lower right")
      fig.set_size_inches(mpl.rcParams["figure.figsize"][0] * 2 *
                          np.array([1, 2 / 5]))
      args.save_fig(cc_cc_plot_fname)


# ---------------------------------------------------------------------------

# --- Plot degree distribution ----------------------------------------------
  degree_fname = "degrees"
  if args.must_write(degree_fname):
    logger.info("Computing degrees")
    graph.compute_transpose()
    graph.compute_degrees()
    # logger.info("Computing reciprocity")
    # graph.compute_reciprocity()
    logger.info("Plotting degrees")
    jp = plots.degrees_jointplot(
        graph,
        ref_artists=args.ref_artists,
        kind="hist",
        bins=42,
        log_marginal=True,
        marginal_kws=dict(linewidth=0,),
        scatter_kws=dict(
            ec="#acc9ee",
        ),  # kendall_tau=True, reciprocity=True, stats_kw=dict(fontsize=7),
        grid=True)
    jp.ax_marg_x.set_yticks(np.power(10, np.arange(1, 6, 2)))
    jp.ax_marg_y.set_xticks(jp.ax_marg_x.get_yticks())
    args.save_fig(degree_fname)
  # ---------------------------------------------------------------------------

  # --- Plot distance probability function ------------------------------------
  neighborhhod_fname = "distances"
  if args.must_write(neighborhhod_fname):
    logger.info("Computing neighbourhood function")
    graph.compute_transpose()
    graph.compute_neighborhood()

    logger.info("Plotting distance probability mass function")
    plots.plot_distances(graph,
                         kind="both",
                         zorder=100,
                         area_kws=dict(alpha=.5),
                         hdi_hs=-0.03)
    plt.grid()
    plt.gca().yaxis.label.set_rotation(0)
    plt.gca().yaxis.label.set_text("relative frequency")
    plt.gca().yaxis.label.set_horizontalalignment("left")
    plt.gca().yaxis.label.set_verticalalignment("bottom")
    plt.gca().yaxis.set_label_coords(-0.06, 1.02)
    plt.gcf().set_size_inches(mpl.rcParams["figure.figsize"] *
                              np.array([1, 9 / 16]))
    # plt.title(f"{graph.basename}\nHyperBall ($log_2m$ = 8)")
    args.save_fig(neighborhhod_fname)
  # ---------------------------------------------------------------------------
