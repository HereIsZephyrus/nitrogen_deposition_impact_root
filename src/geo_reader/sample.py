"""
Raster sampling utilities using GDAL
"""
import logging
from typing import List, Tuple
from math import floor
import numpy as np
from osgeo import gdal
from .raster_reader import Raster
from .sample_reader import Sample

logger = logging.getLogger(__name__)

def sample_raster(raster: Raster, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Sample raster data at given points using GDAL

    Args:
        raster: Raster object to sample from
        points: List of (x, y) coordinates. 

    Returns:
        Array of sampled values with shape (n_points, bands) or (n_points,) for single band
    """
    if not points:
        return None

    if raster.geotransform is None:
        logger.error("Raster must have valid geotransform")
        return None

    # Get inverse geotransform for coordinate conversion
    gt_forward = raster.geotransform
    gt_reverse = gdal.InvGeoTransform(gt_forward)
    if gt_reverse is None:
        logger.error("Cannot compute inverse geotransform")
        return None

    n_bands = raster.data.shape
    n_points = len(points)
    sampled_values = np.full((n_points, n_bands), np.nan, dtype=np.float32)

    px_list = []
    py_list = []
    for (mx, my) in points:
        px, py = gdal.ApplyGeoTransform(gt_reverse, mx, my)
        px = int(floor(px))  # x pixel (column)
        py = int(floor(py))  # y pixel (row)
        px_list.append(px)
        py_list.append(py)

    sampled_values = raster.data[:, py_list, px_list]
    return sampled_values

def get_sample_data(sample_file: str, raster: Raster) -> np.ndarray:
    """
    raster shape in (dim, x, y)
    """
    samples = Sample(sample_file)
    sample_locations = samples.get_location()
    # sample in raster
    sample_data = sample_raster(raster, sample_locations)

    return sample_data
