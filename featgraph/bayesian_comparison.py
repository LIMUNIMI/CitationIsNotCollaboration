"""Utilities for Bayesian comparison of populations"""
from scipy import stats
import numpy as np
import pymc3 as pm
import copy
from typing import Optional


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
               dof_mean: float = 30):
    self.model = model
    self.model_ = None
    self.mean_std_scale = mean_std_scale
    self.std_std_scale = std_std_scale
    self.dof_mean = dof_mean
    self.deepcopy = deepcopy

  def fit(self, x_1, x_2, name_1: str = "group1", name_2: str = "group2"):
    """Fit the comparison model to two (unpaired) population samples

    Args:
      x_1 (array): The first population's samples
      x_2 (array): The second population's samples
      name_1 (str): The first population's name (default is :data:`"group1"`)
      name_2 (str): The second population's name (default is :data:`"group2"`)

    Returns:
      self"""
    self.model_ = pm.Model() if self.model is None else (
        copy.deepcopy(self.model) if self.deepcopy else self.model)

    x = np.empty(len(x_1) + len(x_2))
    x[:len(x_1)] = x_1
    x[len(x_1):] = x_2

    # Paramaters for distributions of means
    m_x, s_x = stats.norm.fit(x)
    s_m_x = self.mean_std_scale * s_x

    # Paramaters for distributions of standard deviations
    s_x_lo = s_x / self.std_std_scale
    s_x_hi = s_x * self.std_std_scale

    with self.model_:
      # Parameters of T distributions
      m_1 = pm.Normal(fr"{name_1} $\mu$", mu=m_x, sd=s_m_x)
      m_2 = pm.Normal(fr"{name_2} $\mu$", mu=m_x, sd=s_m_x)
      s_1 = pm.Uniform(fr"{name_1} $\sigma$", lower=s_x_lo, upper=s_x_hi)
      s_2 = pm.Uniform(fr"{name_2} $\sigma$", lower=s_x_lo, upper=s_x_hi)
      dof = pm.Exponential(r"$\nu - 1$", lam=1 / (self.dof_mean - 1)) + 1

      # Data distributions
      t_1 = pm.StudentT(name_1, nu=dof, mu=m_1, lam=s_1**-2, observed=x_1)  # pylint: disable=W0612
      t_2 = pm.StudentT(name_2, nu=dof, mu=m_2, lam=s_2**-2, observed=x_2)  # pylint: disable=W0612

      # Estimate group differences
      diff_of_means = pm.Deterministic(r"$\Delta\mu$", m_1 - m_2)
      diff_of_stds = pm.Deterministic(r"$\Delta\sigma$", s_1 - s_2)  # pylint: disable=W0612
      effect_size = pm.Deterministic(  # pylint: disable=W0612
          "effect size", diff_of_means / np.sqrt((s_1**2 + s_2**2) / 2))

    return self

  def sample(self, *args, **kwargs):
    """Sample from the fitted model

    Args:
      args: Positional arguments for :func:`pymc3.sample`
      kwargs: Keyword arguments for :func:`pymc3.sample`"""
    with self.model_:
      return pm.sample(*args, **kwargs)
