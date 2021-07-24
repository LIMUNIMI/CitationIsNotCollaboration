"""Plot utilities. This module requires matplotlib"""
from matplotlib import pyplot as plt
import featgraph.jwebgraph.utils
from featgraph import jwebgraph
from featgraph.misc import VectorOrCallable
from typing import Optional


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
  ax.scatter(
    x() if callable(x) else x,
    y() if callable(y) else y,
    **kwargs
  )
  ax.set_xscale(xscale)
  ax.set_yscale(yscale)
  # Kendall Tau
  if kendall_tau:
    kt = jwebgraph.utils.kendall_tau(x, y)
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
    tit_li.append(r"(Kendall $\tau$ = {:.5f})".format(kt))
  if len(tit_li) > 0:
    ax.set_title("\n".join(tit_li))
  return ax
