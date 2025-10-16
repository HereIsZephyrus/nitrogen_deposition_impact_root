from ..variance import Variance

class GridCalculator:
    def __init__(self,mdl):
        self.mdl = mdl

    def calculate_result(self, variance: Variance) -> float:
        return 0.0

    def calculate(self, lat: float, lon: float) -> float:
        variance = self._sample_variance(lat, lon)
        return self.calculate_result(variance)

    def _sample_variance(self, lat: float, lon: float) -> Variance:
        return Variance()
