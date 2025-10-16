from pydantic import BaseModel

class ClimateVariance(BaseModel):
    """
    background variance of climate data
    """
    avgt_temp: float
    dem: float
    # ...

class SoilVariance(BaseModel):
    """
    background variance of soil data
    """
    soil_ph: float
    # ...

class VegetationVariance(BaseModel):
    """
    background variance of vegetation data
    """
    vegetation_type: int # one-hot encoding
    # ...

class NitrogenVariance(BaseModel):
    """
    background variance of nitrogen data
    """
    nitrogen_decomposed: float
    # ...

class Variance(BaseModel):
    climate: ClimateVariance
    soil: SoilVariance
    vegetation: VegetationVariance
    nitrogen: NitrogenVariance