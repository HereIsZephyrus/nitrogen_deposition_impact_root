import logging
from typing import List
import numpy as np
from .nls_model import NLSModel

logger = logging.getLogger(__name__)

class MultiplicativeModel(NLSModel):
    """
    Multiplicative model: nitrogen deposition through the multiplicative effect of PCA components on biomass

    Model form:
    biomass = β0 * exp(Σ(βi * PCi) + Σ(γi * N * PCi))

    Or simplified as:
    biomass = β0 * Π(exp(βi * PCi)) * Π(exp(γi * N * PCi))
    """

    def __init__(self):
        super().__init__("Multiplicative Interaction Model")

    def model_func(self, X: np.ndarray, *params) -> np.ndarray:
        nitrogen = X[0, :]
        pcs = X[1:, :]

        beta0 = params[0]
        n_pcs = pcs.shape[0]
        betas = np.array(params[1:n_pcs+1])
        gammas = np.array(params[n_pcs+1:])

        # Predicted values: biomass = β0 * exp(Σ(βi * PCi) + Σ(γi * N * PCi))
        exponent = np.sum(betas[:, np.newaxis] * pcs, axis=0)
        exponent += np.sum(gammas[:, np.newaxis] * nitrogen[np.newaxis, :] * pcs, axis=0)

        pred = beta0 * np.exp(exponent)

        return pred

    def _get_n_params(self, X: np.ndarray) -> int:
        n_pcs = X.shape[1] - 1
        return 1 + n_pcs + n_pcs

    def get_param_names(self) -> List[str]:
        if self.params is None:
            return []
        n_total = len(self.params)
        n_pcs = (n_total - 1) // 2

        names = ['β0_scale']
        names += [f'β{i+1}_PC{i+1}' for i in range(n_pcs)]
        names += [f'γ{i+1}_N×PC{i+1}' for i in range(n_pcs)]
        return names

    def get_initial_params(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_pcs = X.shape[1] - 1
        p0 = np.zeros(self._get_n_params(X))
        p0[0] = np.median(y)  # Scale parameter
        p0[1:n_pcs+1] = 0.01
        p0[n_pcs+1:] = 0.001
        return p0
