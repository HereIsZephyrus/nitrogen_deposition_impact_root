"""
Raster sampling utilities using GDAL
"""
import logging
from typing import List, Tuple
from math import floor
import numpy as np
from osgeo import gdal
from .raster_reader import Raster

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

    height, width, n_bands = raster.data.shape
    n_points = len(points)
    sampled_values = np.full((n_points, n_bands), np.nan, dtype=np.float32)

    # Sample each point
    for i, (mx, my) in enumerate(points):
        # Convert from map coordinates to pixel coordinates using GDAL
        px, py = gdal.ApplyGeoTransform(gt_reverse, mx, my)
        px = int(floor(px))  # x pixel (column)
        py = int(floor(py))  # y pixel (row)

        # Check if pixel is within bounds
        if 0 <= py < height and 0 <= px < width:
            if n_bands == 1:
                # Single band raster
                sampled_values[i, 0] = raster.data[py, px]
            else:
                # Multi-band raster
                sampled_values[i, :] = raster.data[py, px, :]
        else:
            logger.warning("Point %d (%.6f, %.6f) -> pixel (%d, %d) is out of bounds", i, mx, my, px, py)

    return sampled_values
