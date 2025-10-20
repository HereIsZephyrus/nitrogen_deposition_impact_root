import logging
from typing import List
import numpy as np

from .nls_model import NLSModel

logger = logging.getLogger(__name__)

class LinearModel(NLSModel):
    """
    Linear model (OLS): simple linear regression as baseline reference

    Model form:
    biomass = β0 + β_N * N + Σ(βi * PCi)

    Where:
    - β0: intercept
    - β_N: coefficient for nitrogen addition
    - βi: coefficients for PCA components
    - N: nitrogen addition
    - PCi: i-th principal component

    This is the simplest model using ordinary least squares (OLS) as a baseline for comparison.
    """

    def __init__(self):
        super().__init__("Linear Model (OLS)")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        """
        X: shape (n_features, n_samples)
           X[0, :] = nitrogen_add
           X[1:, :] = PC components
        """
        nitrogen = X[0, :]
        pcs = X[1:, :]

        # Parameter allocation
        # params = [β0, β_N, β1, ..., βn]
        beta0 = params[0]
        beta_n = params[1]
        betas = np.array(params[2:])

        # Calculate predicted values
        # biomass = β0 + β_N * N + Σ(βi * PCi)
        pred = beta0 + beta_n * nitrogen
        pred += np.sum(betas[:, np.newaxis] * pcs, axis=0)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1  # Subtract nitrogen column
        return 2 + n_pcs  # β0 + β_N + βs

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_pcs = len(self.params) - 2

        names = ['β0_intercept', 'β_N']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Use simple linear regression to get initial parameter estimates"""
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))

        # Use mean of y as intercept initial value
        p0[0] = np.mean(y)
        # Initial value for nitrogen coefficient
        p0[1] = 0.0
        # Initial values for PC coefficients
        p0[2:] = 0.0

        return p0
