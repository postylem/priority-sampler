import numpy as np


class PrioritySampler:
    def __init__(self, kind: str = "priority", rng_opts=dict()):
        self.rng = np.random.default_rng(**rng_opts)
        self.kind = kind
        self.perturb = self._perturb[kind]
        self.p_incl = self._p_incl[kind]

    def weighted_sample(self, w: np.ndarray, size: int) -> np.ndarray:
        z = self.perturb(w)
        S, t = self._threshold(z, size)
        return w / self.p_incl(w, t) * np.isin(np.arange(len(w)), S)

    @staticmethod
    def _threshold(z: np.ndarray, k: int):
        """Get lowest k values' indices, and (k+1)-smallest value from array z"""
        sorted_indices = np.argsort(z)
        # print(z, "sorted indices", sorted_indices)
        S, t = sorted_indices[:k], z[sorted_indices[k]]
        # print("smallest k, and thresh", S, t)
        return S, t

    _perturb = {
        "priority": lambda w: np.random.uniform(low=0, high=1 / w, size=len(w)),
        "exp": lambda w: np.random.exponential(scale=1 / w, size=len(w)),
        # "gumbel": lambda w: np.log(w) + np.random.gumbel(loc=0, scale=1, size=len(w)),
    }
    _p_incl = {
        # Pr(i ∈ S | t) = Pr(Uniform(0,1/w) < t) = min(1, wt)
        "priority": lambda w, t: np.minimum(1, w * t),
        # Pr(i ∈ S | t) = Pr(Exponential(scale=1/w) < t)
        "exp": lambda w, t: 1 - np.exp(-t * w),
        # Pr(i ∈ S | t) = Pr(Gumbel(loc=lw) > t) = 1 - Pr(Gumbel(loc=lw) < t)
        # "gumbel": lambda lw, t: 1 - np.exp(-np.exp(lw - t)),
    }
