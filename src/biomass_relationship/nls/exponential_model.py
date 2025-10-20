import logging
from typing import List
import numpy as np
from .nls_model import NLSModel

logger = logging.getLogger(__name__)

class ExponentialModel(NLSModel):
    """
    Exponential model: exponential response model

    Model form:
    biomass = β0 + Σ(αi * PCi) * exp(β * N)

    Or:
    biomass = β0 * exp(Σ(βi * PCi) + γ * N)
    """

    def __init__(self, variant: str = 'v1'):
        """
        Args:
            variant: 'v1' or 'v2'
                v1: Predicted values: biomass = β0 + Σ(αi * PCi) * exp(β * N)
                v2: Predicted values: biomass = β0 * exp(Σ(βi * PCi) + γ * N)
        """
        self.variant = variant
        name = f"Exponential-{variant} Model"
        super().__init__(name)

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]
        n_pcs = pcs.shape[0]

        if self.variant == 'v1':
            # biomass = β0 + Σ(αi * PCi) * exp(β * N)
            beta0 = params[0]
            beta_n = params[1]
            alphas = np.array(params[2:])

            pc_effect = np.sum(alphas[:, np.newaxis] * pcs, axis=0)
            pred = beta0 + pc_effect * np.exp(beta_n * nitrogen)

        else:  # v2
            # biomass = β0 * exp(Σ(βi * PCi) + γ * N)
            beta0 = params[0]
            betas = np.array(params[1:n_pcs+1])
            gamma = params[-1]

            exponent = np.sum(betas[:, np.newaxis] * pcs, axis=0) + gamma * nitrogen
            pred = beta0 * np.exp(exponent)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1
        if self.variant == 'v1':
            return 2 + n_pcs  # β0, β_n, αs
        else:
            return 2 + n_pcs  # β0, βs, γ

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_pcs = len(self.params) - 2

        if self.variant == 'v1':
            names = ['β0_intercept', 'β_N']
            names += [f'α{i+1}_PC{i+1}' for i in range(n_pcs)]
        else:
            names = ['β0_scale']
            names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
            names.append('γ_N')
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))

        if self.variant == 'v1':
            p0[0] = np.mean(y)
            p0[1] = 0.01
            p0[2:] = 0.1
        else:
            p0[0] = np.median(y)
            p0[1:-1] = 0.01
            p0[-1] = 0.001

        return p0