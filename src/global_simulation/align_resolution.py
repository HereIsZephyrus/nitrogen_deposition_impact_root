"""
Align resolution of the global forest data to 1 minutes
"""
import os
import logging
import numpy as np
import xarray as xr
from osgeo import gdal, osr

gdal.UseExceptions()

logger = logging.getLogger(__name__)
from ..geo_reader import Raster

class ResolutionAligner:
    """
    Align resolution of the global forest data to 1 minutes
    """
    def __init__(self, target_resolution: float):
        self.target_resolution = target_resolution

    def align(self, raster: Raster) -> Raster:
        if raster.resolution == self.target_resolution:
            return raster
        
        if raster.resolution > self.target_resolution:
            return self._downsample(raster)
        else:
            return self._upsample(raster)

    def _downsample(self, raster: Raster) -> Raster:
        pass

    def _upsample(self, raster: Raster) -> Raster:
        pass
