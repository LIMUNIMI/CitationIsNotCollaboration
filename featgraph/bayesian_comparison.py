"""Utilities for Bayesian comparison of populations"""
from scipy import stats
import numpy as np
import pymc3 as pm
import arviz
import copy
from typing import Optional, Tuple


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
