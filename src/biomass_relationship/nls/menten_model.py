import logging
from typing import List
import numpy as np
from .nls_model import NLSModel

logger = logging.getLogger(__name__)

class MichaelisMentenModel(NLSModel):
    """
    Michaelis-Menten model: saturation response model

    Model form:
    biomass = (Vmax * f(PC)) / (Km + N) + baseline

    Where:
    - Vmax: maximum response
    - Km: half-saturation constant
    - f(PC): linear combination of PCA components
    - baseline: baseline biomass
    """

    def __init__(self):
        super().__init__("Michaelis-Menten Saturation Model")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]

        # Parameters: [Vmax, Km, baseline, β1, ..., βn]
        Vmax = params[0]
        Km = params[1]
        baseline = params[2]
        betas = np.array(params[3:])

        # f(PC) = Σ(βi * PCi) = linear combination of PCA components
        f_pc = np.sum(betas[:, np.newaxis] * pcs, axis=0)

        # Predicted values: biomass = (Vmax * f(PC)) / (Km + N) + baseline
        pred = (Vmax * f_pc) / (Km + nitrogen + 1e-10) + baseline

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1
        return 3 + n_pcs  # Vmax, Km, baseline, βs = parameters

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_pcs = len(self.params) - 3

        names = ['Vmax', 'Km', 'baseline']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))
        p0[0] = np.max(y) - np.min(y)  # Vmax
        p0[1] = np.mean(X[:, 0])  # Km
        p0[2] = np.min(y)  # baseline
        p0[3:] = 0.1  # βs
        return p0
