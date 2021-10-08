"""Utilities for Bayesian comparison of populations"""
from featgraph import report
from scipy import stats
import pandas as pd
import numpy as np
import pymc3 as pm
import copy
from typing import Optional, Tuple, Sequence, List


class StudentTComparison:
  """Bayesian comparison model for Student's T-distributed datasets

  Args:
    model (:class:`pymc3.Model`): Starting model to use (optional)
    deepcopy (bool): If :data:`True` (default), then deepcopy
      the given :data:`model`, otherwise use it as-is
    mean_std_scale (float): Scaling factor for the standard deviation of means
    std_std_scale (float): Scaling factor for the boundaries of the standard
      deviations distributions
    dof_mean (float): Prior expectation for the T-distribution
      degrees of freedom"""

  def __init__(self,
               model: Optional = None,
               deepcopy: bool = True,
               mean_std_scale: float = 2.0,
               std_std_scale: float = 1000.0,
               dof_mean: float = 30.0):
    self.model = model
    self.model_ = None
    self.mean_std_scale = mean_std_scale
    self.std_std_scale = std_std_scale
    self.dof_mean = dof_mean
    self.deepcopy = deepcopy

  def fit(self, **kwargs):
    """Fit the comparison model to two (unpaired) population samples

    Args:
      kwargs: Named arrays of population data

    Returns:
      self"""
    self.model_ = pm.Model() if self.model is None else (
        copy.deepcopy(self.model) if self.deepcopy else self.model)

    x = np.empty(sum(map(len, kwargs.values())))
    i = 0
    j = 0
    for xi in kwargs.values():
      i = j
      j += len(xi)
      x[i:j] = xi

    # Paramaters for distributions of means
    m_x, s_x = stats.norm.fit(x)
    del x
    s_m_x = self.mean_std_scale * s_x

    # Paramaters for distributions of standard deviations
    s_x_lo = s_x / self.std_std_scale
    s_x_hi = s_x * self.std_std_scale

    with self.model_:
      # Parameters of T distributions
      means = {k: pm.Normal(f"{k} mean", mu=m_x, sd=s_m_x) for k in kwargs}
      stds = {
          k: pm.Uniform(f"{k} std", lower=s_x_lo, upper=s_x_hi) for k in kwargs
      }
      dof = pm.Exponential("dof - 1", lam=1 / (self.dof_mean - 1)) + 1

      # Data distributions
      student_ts = {  # pylint: disable=W0612
          k: pm.StudentT(k, nu=dof, mu=means[k], lam=stds[k]**-2, observed=v)
          for k, v in kwargs.items()
      }

      # Estimate group differences
      diff_of_means = {
          k: {
              h: pm.Deterministic(f"mean {k} - {h}", means[k] - means[h])
              for h in kwargs
              if h != k
          } for k in kwargs
      }
      diff_of_stds = {  # pylint: disable=W0612
          k: {
              h: pm.Deterministic(f"std {k} - {h}", stds[k] - stds[h])
              for h in kwargs
              if h != k
          } for k in kwargs
      }
      effect_sizes = {  # pylint: disable=W0612
          k: {
              h: pm.Deterministic(
                  f"effect size {k} - {h}", diff_of_means[k][h] / np.sqrt(
                      (stds[k]**2 + stds[h]**2) / 2)) for h in kwargs if h != k
          } for k in kwargs
      }

    return self

  def sample(self, *args, **kwargs):
    """Sample from the fitted model

    Args:
      args: Positional arguments for :func:`pymc3.sample`
      kwargs: Keyword arguments for :func:`pymc3.sample`"""
    with self.model_:
      return pm.sample(*args, **kwargs)


def rope_probabilities(
    data,
    key: str = "posterior",
    var: str = "effect size",
    rope: Tuple[float, float] = (-0.1, 0.1)
) -> Tuple[float, float, float]:
  """Return the probabilities that a variable in the data is below,
  within or above a ROPE

  Args:
    data: Inference data
    key (str): Key for data values. Default is :data:`"posterior"`
    var (str): variable name. Default is :data:`"effect size"`
    rope (copule of float): The boundaries of the ROPE.
      Default is :data:`(-0.1, 0.1)`

  Returns:
    triplet of float: The probabilities that the variable is below,
    within or above the ROPE"""
  v = data[key][var]
  p_lt = (v < rope[0]).mean().values[()]
  p_in = ((rope[0] <= v) & (v <= rope[1])).mean().values[()]
  p_gt = (v > rope[1]).mean().values[()]
  return p_lt, p_in, p_gt


def _rope_probabilities_names(
    df: pd.DataFrame,
    names: Optional[Sequence[str]] = None,
    order=None,
    x_label: str = "x",
    y_label: str = "y",
    probs_labels: Tuple[str, str, str] = (
        "x - y < ROPE",
        "x - y in ROPE",
        "x - y > ROPE",
    )
) -> List[str]:
  """Get the population names from a ROPE probabilities dataframe

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

  Returns:
    list of str: The list of names"""
  if names is None:
    names = pd.concat((df[x_label], df[y_label])).unique().tolist()
  if order is not None:
    if order == "auto":
      # Order by rank
      score = df.groupby(by=[x_label]).mean()
      score = score[probs_labels[2]] - score[probs_labels[0]]
      order = score.sort_values().iloc[::-1].index.to_list().index
    names = sorted(names, key=order)
  return names


def make_report(df: pd.DataFrame,
                metric: str,
                th: float = 0.05,
                rope: Optional[Tuple[float, float]] = (-0.1, 0.1),
                rope_var: str = "effect size",
                names: Optional[Sequence[str]] = None,
                order=None,
                x_label: str = "x",
                y_label: str = "y",
                probs_labels: Tuple[str, str, str] = (
                    "x - y < ROPE",
                    "x - y in ROPE",
                    "x - y > ROPE",
                ),
                **kwargs) -> report.ReportMaker:
  """Build a report from a ROPE probabilities dataframe

  Args:
    df (pd.DataFrame): ROPE probabilities dataframe
    metric (str): The name of the evaluated metric
    th (float): The threshold for hypothesis rejection
    rope (couple of float): The boundaries of the ROPE. If :data:`None`,
      don't define the ROPE in the report. Default is :data:`(-0.1, 0.1)`
    rope_var (str): Name of the random variable on which the ROPE is defined.
       Default is :data:`"effect size"
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
    type (str): Type of report to output. See
      :class:`featgraph.report.ReportMaker` for a list of available types"""
  names = _rope_probabilities_names(
      df=df,
      names=names,
      order=order,
      x_label=x_label,
      y_label=y_label,
      probs_labels=probs_labels,
  )
  rm = report.ReportMaker(type=kwargs.get("type", "plain-text"))
  rm.append(f"We compared the {metric} values across {len(names)} ",
            "populations using BEST.\n")
  if rope is not None:
    rm.append(f"We defined a ROPE over the {rope_var}, as "
              f"the interval between {rope[0]} and {rope[1]}.\n")
  rm.append("For every comparison, we make three hypoteses: ",
            f"the {rope_var} is either below the ROPE, ",
            "within the ROPE, or above the ROPE. ",
            "An hypothesis is accepted only if the posterior ",
            "probability of the others ",
            f"is below the threshold of {th * 100:.2f}")
  rm.append("%", escape=True)
  rm.append(".")

  with rm.itemize() as it:
    for n in names:
      data_n = df[df[x_label] == n]
      n_gt = data_n[data_n[probs_labels[2]] >= (1 - th)][y_label]
      n_pe = data_n[data_n[probs_labels[1]] >= (1 - th)][y_label]
      n_lt = data_n[data_n[probs_labels[0]] >= (1 - th)][y_label]

      def prob_n(other: str, prob_idx: int, d=data_n) -> float:
        return d[d[y_label] == other][probs_labels[prob_idx]].mean() * 100

      n_of_cases = (len(n_gt) > 0) + (len(n_pe) > 0) + (len(n_lt) > 0)
      if n_of_cases == 0:
        continue
      it.item(n)
      with it.itemize() as iit:
        n_ = "\"" + n + "\""
        if len(n_gt) > 0:
          iit.item(f"the {rope_var} is above the ROPE for tests ",
                   f"between population {n_} and populations")
          with iit.itemize() as iiit:
            for m in n_gt:
              iiit.item(m)
              iiit.append(f" ({prob_n(m, 2):.2f}%)", escape=True)

        if len(n_pe) > 0:
          iit.item(f"the {rope_var} is within the ROPE for tests ",
                   f"between population {n_} and populations")
          with iit.itemize() as iiit:
            for m in n_pe:
              iiit.item(m)
              iiit.append(f" ({prob_n(m, 1):.2f}%)", escape=True)

        if len(n_lt) > 0:
          iit.item(f"the {rope_var} is below the ROPE for tests ",
                   f"between population {n_} and populations")
          with iit.itemize() as iiit:
            for m in n_lt:
              iiit.item(m)
              iiit.append(f" ({prob_n(m, 0):.2f}%)", escape=True)
  return rm
