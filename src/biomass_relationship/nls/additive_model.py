import logging
from typing import List
import numpy as np
from .nls_model import NLSModel

logger = logging.getLogger(__name__)

class AdditiveModel(NLSModel):
    """
    Additive model: nitrogen deposition through the additive effect of PCA components on biomass

    Model form:
    biomass = β0 + Σ(βi * PCi) + Σ(γi * N * PCi)

    Where:
    - β0: intercept
    - βi: main effect of PCA components
    - γi: interaction effect of nitrogen deposition and PCA components
    - N: nitrogen addition
    - PCi: i-th principal component
    """

    def __init__(self):
        super().__init__("Additive Interaction Model")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        """
        X: shape (n_features, n_samples)
           X[0, :] = nitrogen_add
           X[1:, :] = PC components
        """
        nitrogen = X[0, :]
        pcs = X[1:, :]
        n_pcs = pcs.shape[0]

        # Parameter allocation
        # params = [β0, β1, ..., βn, γ1, ..., γn]
        beta0 = params[0]
        betas = np.array(params[1:n_pcs+1])
        gammas = np.array(params[n_pcs+1:])

        # Calculate predicted values
        # biomass = β0 + Σ(βi * PCi) + Σ(γi * N * PCi)
        pred = beta0
        pred += np.sum(betas[:, np.newaxis] * pcs, axis=0)
        pred += np.sum(gammas[:, np.newaxis] * nitrogen[np.newaxis, :] * pcs, axis=0)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1  # Subtract nitrogen column
        return 1 + n_pcs + n_pcs  # β0 + βs + γs

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_total = len(self.params)
        n_pcs = (n_total - 1) // 2

        names = ['β0_intercept']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        names += [f'γ{i+1}_N×PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))
        p0[0] = np.mean(y)  # Initial value of intercept is the mean of y
        p0[1:n_pcs+1] = 0.1  # Initial value of main effect
        p0[n_pcs+1:] = 0.01  # Initial value of interaction effect (smaller)
        return p0
