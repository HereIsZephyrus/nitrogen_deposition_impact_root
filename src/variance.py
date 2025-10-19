"""
Variance models for environmental data based on sample CSV structure
"""
from typing import List
from pydantic import BaseModel, Field
import pandas as pd

class ClimateVariance(BaseModel):
    """
    Climate variance data based on WorldClim bioclimatic variables (BIO1-BIO19)
    Reference: https://worldclim.org/data/bioclim.html
    """
    bio1: float = Field(..., description="BIO1: Annual Mean Temperature (℃)")
    bio12: float = Field(..., description="BIO12: Annual Precipitation (mm)")

    bio2: float = Field(..., description="BIO2: Mean Diurnal Range (℃)")
    bio3: float = Field(..., description="BIO3: Isothermality (BIO2/BIO7 × 100)")
    bio4: float = Field(..., description="BIO4: Temperature Seasonality (standard deviation × 100)")
    bio5: float = Field(..., description="BIO5: Max Temperature of Warmest Month (℃)")
    bio6: float = Field(..., description="BIO6: Min Temperature of Coldest Month (℃)")
    bio7: float = Field(..., description="BIO7: Temperature Annual Range (BIO5-BIO6) (℃)")

    bio8: float = Field(..., description="BIO8: Mean Temperature of Wettest Quarter (℃)")
    bio9: float = Field(..., description="BIO9: Mean Temperature of Driest Quarter (℃)")
    bio10: float = Field(..., description="BIO10: Mean Temperature of Warmest Quarter (℃)")
    bio11: float = Field(..., description="BIO11: Mean Temperature of Coldest Quarter (℃)")

    bio13: float = Field(..., description="BIO13: Precipitation of Wettest Month (mm)")
    bio14: float = Field(..., description="BIO14: Precipitation of Driest Month (mm)")
    bio15: float = Field(..., description="BIO15: Precipitation Seasonality (Coefficient of Variation)")
    bio16: float = Field(..., description="BIO16: Precipitation of Wettest Quarter (mm)")
    bio17: float = Field(..., description="BIO17: Precipitation of Driest Quarter (mm)")
    bio18: float = Field(..., description="BIO18: Precipitation of Warmest Quarter (mm)")
    bio19: float = Field(..., description="BIO19: Precipitation of Coldest Quarter (mm)")
    elevation: float = Field(..., description="Elevation (m)")


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

    def model_dump(self, *, only_continuous: bool = False, **kwargs) -> dict:
        """
        Dump model to dictionary

        Args:
            only_continuous: If True, exclude categorical variables
            **kwargs: Additional arguments passed to pydantic's model_dump

        Returns:
            Dictionary with all variables (or only continuous if only_continuous=True)
        """
        climate_data = self.climate.model_dump(**kwargs)
        soil_data = self.soil.model_dump(**kwargs)
        vegetation_data = self.vegetation.model_dump(**kwargs)
        nitrogen_data = self.nitrogen.model_dump(**kwargs)

        if only_continuous:
            # Remove categorical variables
            # VegetationVariance: all fields are categorical
            # NitrogenVariance: fertilizer_type and treatment_date are categorical
            categorical_fields = {
                'vegetation_type',  # VegetationVariance
                'fertilizer_type',  # NitrogenVariance
                'treatment_date'    # NitrogenVariance
            }

            # Filter out categorical variables
            vegetation_data = {}  # All vegetation fields are categorical
            nitrogen_data = {k: v for k, v in nitrogen_data.items() if k not in categorical_fields}

        return {
            **climate_data,
            **soil_data,
            **vegetation_data,
            **nitrogen_data
        }

def unpack_variance(input_var: List[Variance]) -> pd.DataFrame:
    """
    Unpack variance into a dataframe
    """
    data = [var.model_dump(only_continuous = True) for var in input_var]
    return pd.DataFrame(data)
