"""
Variance models for environmental data based on sample CSV structure
"""
from pydantic import BaseModel, Field
from typing import Optional


class ClimateVariance(BaseModel):
    """
    Climate variance data - precipitation and temperature
    """
    p: float = Field(..., description="Precipitation (mm/yr)")
    ta: float = Field(..., description="Average temperature (℃)")


class SoilVariance(BaseModel):
    """
    Soil property variance data (20 properties)
    """
    # Water and texture properties
    available_water_capacity_for_rootable_soil_depth: float = Field(..., description="Available water capacity")
    coarse_fragments: float = Field(..., description="Coarse fragments (%)")
    sand: float = Field(..., description="Sand content (%)")
    silt: float = Field(..., description="Silt content (%)")
    clay: float = Field(..., description="Clay content (%)")
    bulk: float = Field(..., description="Bulk density (g/cm³)")

    # Chemical properties
    organic_carbon_content: float = Field(..., description="Organic carbon content (%)")
    ph_in_water: float = Field(..., description="pH in water")
    total_nitrogen_content: float = Field(..., description="Total nitrogen content (%)")
    cn_ratio: float = Field(..., description="C:N ratio")

    # Cation exchange capacity
    cec_soil: float = Field(..., description="CEC of soil (cmol/kg)")
    cec_clay: float = Field(..., description="CEC of clay (cmol/kg)")
    cec_eff: float = Field(..., description="Effective CEC (cmol/kg)")

    # Base saturation and other properties
    teb: float = Field(..., description="Total exchangeable bases (cmol/kg)")
    bsat: float = Field(..., description="Base saturation (%)")
    alum_sat: float = Field(..., description="Aluminum saturation (%)")
    esp: float = Field(..., description="Exchangeable sodium percentage (%)")
    tcarbon_eq: float = Field(..., description="Total carbon equivalent (%)")
    gypsum: float = Field(..., description="Gypsum content (%)")
    elec_cond: float = Field(..., description="Electrical conductivity (dS/m)")


class VegetationVariance(BaseModel):
    """
    Vegetation type variance data
    """
    vegetation_type: int = Field(..., description="Vegetation type (categorical code: 0=forest, 1=grassland)")


class NitrogenVariance(BaseModel):
    """
    Nitrogen treatment variance data (4 features)
    """
    n_addition: float = Field(..., description="N addition rate (kg N ha⁻¹ y⁻¹)")
    fertilizer_type: int = Field(..., description="Fertilizer type (categorical code)")
    treatment_date: int = Field(..., description="Treatment date (categorical code)")
    duration: float = Field(..., description="Duration (years)")


class Variance(BaseModel):
    """
    Complete variance model combining all environmental factors
    """
    climate: ClimateVariance
    soil: SoilVariance
    vegetation: VegetationVariance
    nitrogen: NitrogenVariance