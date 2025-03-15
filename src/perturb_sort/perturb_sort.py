import numpy as np
from scipy.stats import uniform, expon, beta, gumbel_r, gumbel_l
from typing import Dict, Callable, Tuple, Union, List, Any
from .utils import U


class PerturbSort:
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
                "priority": lambda x: uniform(loc=0, scale=1 / x),
                "betamax": lambda x: beta(a=x, b=1),
                "expmin": lambda x: expon(scale=1 / x),
                "gumbelmin": lambda x: gumbel_l(loc=-np.log(x), scale=1),
                "gumbelmax": lambda x: gumbel_r(loc=np.log(x), scale=1),
            }[noiser]
            self.perturb = lambda x: perturb_dist(x).rvs(len(x))
            F = lambda d, t: (d.sf(t) if noiser[-3:] == "max" else d.cdf(t))
            self.cond_incl_pr = lambda t: lambda x: F(perturb_dist(x), t)

    def weighted_sample(self, x: np.ndarray, size: int) -> np.ndarray:
        z = self.perturb(x)
        S, t = self._threshold(z, size)
        return np.isin(np.arange(len(x)), S) * x / self.cond_incl_pr(t)(x)

    _perturb = {
        # "priority": lambda x: np.random.uniform(low=0, high=1 / x, size=len(x)),
        "priority": lambda x: U(len(x)) / x,
        # "betamax": lambda x: np.random.beta(a=w, b=1, size=len(x)),
        "betamax": lambda x: U(len(x)) ** (1 / x),
        # "expmin": lambda x: np.random.exponential(scale=1 / w, size=len(x)),
        "expmin": lambda x: -np.log(U(len(x))) / x,
        # "gumbelmin": lambda x: -np.random.gumbel(loc=np.log(w), scale=1, size=len(x)),
        "gumbelmin": lambda x: -np.log(x) + np.log(-np.log(U(len(x)))),
        # "gumbelmax": lambda x: np.random.gumbel(loc=np.log(w), scale=1, size=len(x)),
        "gumbelmax": lambda x: np.log(x) - np.log(-np.log(U(len(x)))),
    }

    _cond_incl_pr = {
        # Pr(i ∈ S | t) = Pr(Uniform(0,1/x) < t) = min(1, xt)
        "priority": lambda t: lambda x: np.minimum(1, x * t),
        # Pr(i ∈ S | t) = Pr(Beta(x,1) > t) = 1 - Pr(Beta(x,1) < t) = 1 - t^x
        "betamax": lambda t: lambda x: 1 - t**x,
        # Pr(i ∈ S | t) = Pr(Exponential(scale=1/x) < t)
        # "expmin": lambda t: lambda x: 1 - np.exp(-t * x),
        "expmin": lambda t: lambda x: -np.expm1(-t * x),
        # Pr(i ∈ S | t) = Pr(Gumbel_l(loc=-lx) < t) ## Note, this must be the "left-skewed" gumbel!
        # "gumbel": lambda t: lambda x: 1 - np.exp(-np.exp(t + np.log(x))),
        "gumbelmin": lambda t: lambda x: -np.expm1(-x * np.exp(t)),
        # Pr(i ∈ S | t) = Pr(Gumbel(loc=lx) > t) = 1-Pr(Gumbel(loc=lx) < t)
        # "gumbelmax": lambda t: lambda x: 1 - np.exp(-np.exp(np.log(x) - t)),
        "gumbelmax": lambda t: lambda x: -np.expm1(-x * np.exp(-t)),
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
