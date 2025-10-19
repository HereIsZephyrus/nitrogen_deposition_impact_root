from pydantic import BaseModel, Field

class ImpactedSoilVariance(BaseModel):
    """
    Impact soil property variance data (2 key properties)
    """
    soil_ph: float = Field(..., description="Soil pH")
    total_nitrogen: float = Field(..., description="Total nitrogen (g/kg)")

    def __add__(self, other: 'ImpactedSoilVariance') -> 'ImpactedSoilVariance':
        """Add two ImpactedSoilVariance instances"""
        return ImpactedSoilVariance(
            soil_ph=self.soil_ph + other.soil_ph,
            total_nitrogen=self.total_nitrogen + other.total_nitrogen
        )

    def __sub__(self, other: 'ImpactedSoilVariance') -> 'ImpactedSoilVariance':
        """Subtract two ImpactedSoilVariance instances"""
        return ImpactedSoilVariance(
            soil_ph=self.soil_ph - other.soil_ph,
            total_nitrogen=self.total_nitrogen - other.total_nitrogen
        )
