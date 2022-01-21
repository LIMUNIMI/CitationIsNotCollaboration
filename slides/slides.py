"""Make the plots for the slides"""
from featgraph import scriptutils, plots, logger
from chromatictools import cli
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import sys
import os


@cli.main(__name__, *sys.argv[1:])
def main(*argv):
  """Make the plots for the slides"""
  parser = scriptutils.FeatgraphPlotsArgParse(style_file=os.path.join(
      os.path.dirname(__file__), "slides.mplstyle"),
                                              description=__doc__)
  args = parser.custom_parse(argv)
  graph = args.graph

  # Plot degrees
  degree_scatterplot_fname = "degrees.png"
  if args.must_write(degree_scatterplot_fname):
    logger.info("Computing degrees")
    graph.compute_degrees()
    logger.info("Plotting degrees")
    plots.degrees_scatterplot(graph=graph,
                              ref_artists=args.ref_artists,
                              ref_kwargs={"legend_kw": {
                                  "loc": "upper left"
                              }})
    args.save_fig(degree_scatterplot_fname)

  # Plot distance probability functions
  neighborhhod_fname = "distances.svg"
  if args.must_write(neighborhhod_fname):
    logger.info("Computing transposed graph")
    graph.compute_transpose()
    logger.info("Computing neighbourhood function")
    graph.compute_neighborhood()

    logger.info("Plotting distance probability mass function")
    plots.plot_distances(graph)
    plt.title(f"{graph.basename}\nHyperBall ($log_2m$ = 8)")
    args.save_fig(neighborhhod_fname)

  # Violin plots by genre
  for name in args.centrality_specs:
    fname = f"{name}.svg"
    if args.must_write(fname):
      logger.info("Computing %s", name)
      getattr(graph, f"compute_{name}")()
      df, k, violin_order = args.preprocessed_dataset(graph, name)

      logger.info("Plotting %s dataset", name)
      plt.xticks(rotation=33)
      vpl = sns.violinplot(data=df, x="genre", y=k, order=violin_order, cut=0)
      # Remove violin borders
      for c in vpl.collections:
        c.set_linewidth(0)
      plt.gcf().set_size_inches(mpl.rcParams["figure.figsize"][1] *
                                np.array([16 / 9, 1]))
      args.save_fig(fname)

  # Transition line plots
  transition_plot_fname = "transition-{}.svg"
  df, plot_comparison = parser.perform_threshold_comparison(
      args, transition_plot_fname.format)
  if plot_comparison:
    for name, (_, _, norm, logy, k) in args.centrality_specs.items():
      if args.must_write(transition_plot_fname.format(name)):
        logger.info("Plotting %s", k)
        fig, ax = plt.subplots(1, 2, sharex=True)
        fig.set_size_inches(mpl.rcParams["figure.figsize"][1] *
                            np.array([20 / 9, 1]))
        plots.plot_centrality_transitions(df,
                                          k,
                                          median=False,
                                          norm=norm,
                                          logy=logy,
                                          fill_alpha=0.1,
                                          cmap={
                                              "celebrities": "C0",
                                              "community leaders": "C1",
                                              "masses": "C2",
                                              "hip-hop": "C0",
                                              "classical": "C1",
                                              "rock": "C2"
                                          },
                                          fig=fig,
                                          ax=ax)
        args.save_fig(transition_plot_fname.format(name))
