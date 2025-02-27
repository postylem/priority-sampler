import numpy as np
from scipy.stats import uniform, expon, gumbel_r, gumbel_l


class PrioritySampler:
    def __init__(self, kind: str = "uniform", use_scipy=False):
        self.kind = kind
        if not use_scipy:  # faster
            self.perturb = self._perturb[kind]
            self.q = self._q[kind]
        else:  # pedagogical-er
            noise = {
                "uniform": lambda w: uniform(loc=0, scale=1 / w),
                "exponential": lambda w: expon(scale=1 / w),
                "gumbel": lambda w: gumbel_l(loc=-np.log(w), scale=1),
                "gumbelmax": lambda w: gumbel_r(loc=np.log(w), scale=1),
            }[kind]
            self.perturb = lambda w: noise(w).rvs(len(w))
            self.q = {
                # Pr(i ∈ S | t) = Pr(Uniform(0,1/w) < t) = min(1, wt)
                "uniform": lambda w, t: noise(w).cdf(t),
                # Pr(i ∈ S | t) = Pr(Exponential(scale=1/w) < t)
                "exponential": lambda w, t: noise(w).cdf(t),
                # Pr(i ∈ S | t) = Pr(Gumbel(loc=-lw) < t)
                "gumbel": lambda w, t: noise(w).cdf(t),
                # Pr(i ∈ S | t) = Pr(Gumbel(loc=lw) > t) = 1-Pr(Gumbel(loc=lw) < t)
                "gumbelmax": lambda w, t: noise(w).sf(t),
            }[kind]

    def weighted_sample(self, w: np.ndarray, size: int) -> np.ndarray:
        z = self.perturb(w)
        S, t = self._threshold(z, size)
        return w / self.q(w, t) * np.isin(np.arange(len(w)), S)

    _perturb = {
        "uniform": lambda w: np.random.uniform(low=0, high=1 / w, size=len(w)),
        "exponential": lambda w: np.random.exponential(scale=1 / w, size=len(w)),
        "gumbel": lambda w: -np.random.gumbel(loc=np.log(w), scale=1, size=len(w)),
        "gumbelmax": lambda w: np.random.gumbel(loc=np.log(w), scale=1, size=len(w)),
    }

    _q = {
        # Pr(i ∈ S | t) = Pr(Uniform(0,1/w) < t) = min(1, wt)
        "uniform": lambda w, t: np.minimum(1, w * t),
        # Pr(i ∈ S | t) = Pr(Exponential(scale=1/w) < t)
        # "exponential": lambda w, t: 1 - np.exp(-t * w),
        "exponential": lambda w, t: -np.expm1(-t * w),
        # Pr(i ∈ S | t) = Pr(Gumbel_l(loc=-lw) < t) ## Note, this must be the "left-skewed" gumbel!
        # "gumbel": lambda w, t: 1 - np.exp(-np.exp(t + np.log(w))),
        "gumbel": lambda w, t: -np.expm1(-w * np.exp(t)),
        # Pr(i ∈ S | t) = Pr(Gumbel(loc=lw) > t) = 1-Pr(Gumbel(loc=lw) < t)
        # "gumbelmax": lambda w, t: 1 - np.exp(-np.exp(np.log(w) - t)),
        "gumbelmax": lambda w, t: -np.expm1(-w * np.exp(-t)),
    }

    # @staticmethod
    def _threshold(self, z: np.ndarray, k: int):
        """
        Get lowest k values' indices, and (k+1)-smallest value from array z
        (or highest                 , and (k+1)-largest )
        """
        sorted_indices = (
            np.argsort(z) if self.kind[-3:] != "max" else np.argsort(z)[::-1]
        )
        S, t = sorted_indices[:k], z[sorted_indices[k]]
        return S, t
