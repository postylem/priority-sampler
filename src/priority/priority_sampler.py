import numpy as np
from scipy.stats import uniform, expon, beta, gumbel_r, gumbel_l
from typing import Dict, Callable, Tuple, Union, List, Any
from .utils import U


class PrioritySampler:
    noiser: str
    perturb: Callable[[np.ndarray], np.ndarray]
    q: Callable[[np.ndarray, float], np.ndarray]

    def __init__(self, noiser: str = "uniform", use_scipy: bool = False) -> None:
        self.noiser = noiser
        if not use_scipy:  # faster
            self.perturb = self._perturb[noiser]
            self.cond_incl_pr = self._cond_incl_pr[noiser]
        else:  # slower but pedagogical-er
            perturb_dist = {
                "priority": lambda w: uniform(loc=0, scale=1 / w),
                "betamax": lambda w: beta(a=w, b=1),
                "expmin": lambda w: expon(scale=1 / w),
                "gumbelmin": lambda w: gumbel_l(loc=-np.log(w), scale=1),
                "gumbelmax": lambda w: gumbel_r(loc=np.log(w), scale=1),
            }[noiser]
            self.perturb = lambda w: perturb_dist(w).rvs(len(w))
            F = lambda d, t: (d.sf(t) if noiser[-3:] == "max" else d.cdf(t))
            self.cond_incl_pr = lambda t: lambda w: F(perturb_dist(w), t)

    def weighted_sample(self, w: np.ndarray, size: int) -> np.ndarray:
        z = self.perturb(w)
        S, t = self._threshold(z, size)
        return np.isin(np.arange(len(w)), S) * w / self.cond_incl_pr(t)(w)

    # _perturb = {
    #     "priority": lambda w: np.random.uniform(low=0, high=1 / w, size=len(w)),
    #     "betamax": lambda w: np.random.beta(a=w, b=1, size=len(w)),
    #     "exponential": lambda w: np.random.exponential(scale=1 / w, size=len(w)),
    #     "gumbel": lambda w: -np.random.gumbel(loc=np.log(w), scale=1, size=len(w)),
    #     "gumbelmax": lambda w: np.random.gumbel(loc=np.log(w), scale=1, size=len(w)),
    # }

    _perturb = {
        "priority": lambda w: U(len(w)) / w,
        "betamax": lambda w: U(len(w)) ** (1 / w),
        "expmin": lambda w: -np.log(U(len(w))) / w,
        "gumbelmin": lambda w: -np.log(w) + np.log(-np.log(U(len(w)))),
        "gumbelmax": lambda w: np.log(w) - np.log(-np.log(U(len(w)))),
    }

    _cond_incl_pr = {
        # Pr(i ∈ S | t) = Pr(Uniform(0,1/w) < t) = min(1, wt)
        "priority": lambda t: lambda w: np.minimum(1, w * t),
        # Pr(i ∈ S | t) = Pr(Beta(w,1) > t) = 1 - Pr(Beta(w,1) < t) = 1 - t^w
        "betamax": lambda t: lambda w: 1 - t**w,
        # Pr(i ∈ S | t) = Pr(Exponential(scale=1/w) < t)
        # "expmin": lambda t: lambda w: 1 - np.exp(-t * w),
        "expmin": lambda t: lambda w: -np.expm1(-t * w),
        # Pr(i ∈ S | t) = Pr(Gumbel_l(loc=-lw) < t) ## Note, this must be the "left-skewed" gumbel!
        # "gumbel": lambda t: lambda w: 1 - np.exp(-np.exp(t + np.log(w))),
        "gumbelmin": lambda t: lambda w: -np.expm1(-w * np.exp(t)),
        # Pr(i ∈ S | t) = Pr(Gumbel(loc=lw) > t) = 1-Pr(Gumbel(loc=lw) < t)
        # "gumbelmax": lambda t: lambda w: 1 - np.exp(-np.exp(np.log(w) - t)),
        "gumbelmax": lambda t: lambda w: -np.expm1(-w * np.exp(-t)),
    }

    def _threshold(self, z: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
        """
        Get lowest k values' indices, and (k+1)-smallest value from array z
        (or highest                 , and (k+1)-largest )
        """
        idxs = np.argsort(z)
        if self.noiser[-3:] == "max":
            idxs = idxs[::-1]
        sample = idxs[:k]
        threshold = z[idxs[k]]
        return sample, threshold
