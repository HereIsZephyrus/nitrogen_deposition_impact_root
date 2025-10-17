from pydantic import BaseModel, Field
from typing import Optional

class ImpactedSoilVariance(BaseModel):
    """
    Impact soil property variance data (20 properties)
    """
    soil_ph: float = Field(..., description="Soil pH")
    ammonium_nitrogen: float = Field(..., description="Ammonium nitrogen")
    nitrate_nitrogen: float = Field(..., description="Nitrate nitrogen")
    inorganic_nitrogen: float = Field(..., description="Inorganic nitrogen")
    total_nitrogen: float = Field(..., description="Total nitrogen")
