from .calculator import SoilCalculator, transfer_N_input
from .variance import ImpactedSoilVariance

def predict_soil_change(calculator: SoilCalculator, n_addition: float, n_type: str) -> ImpactedSoilVariance:
    """
    Predict soil change for a given nitrogen addition

    Args:
        calculator: SoilCalculator instance
        n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹
        n_type: Nitrogen type (N, NO3, NH4)
    Returns:
        ImpactedSoilVariance instance
    """
    return calculator.predict_all(transfer_N_input(n_addition, n_type))
