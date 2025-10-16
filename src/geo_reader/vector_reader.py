"""
Vector data reader to create mask from vector file and raster
"""
import os
import logging
import numpy as np
from osgeo import gdal, ogr
from .raster_reader import Raster

gdal.UseExceptions()

logger = logging.getLogger(__name__)

def create_mask_location(vector_file: str, raster: Raster) -> np.ndarray:
    """
    Create mask location from vector file and raster using rasterization (faster for large datasets)

    Args:
        vector_file: Path to vector file (shapefile, geojson, etc.)
        raster: Raster object containing data and geotransform

    Returns:
        Boolean mask array, True for pixels inside vector geometry
    """
    if not os.path.exists(vector_file):
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    logger.info("Creating mask using rasterization from: %s", vector_file)
    try:
        vector_ds = ogr.Open(vector_file)
        if vector_ds is None:
            raise ValueError("Cannot open vector file: %s", vector_file)

        layer = vector_ds.GetLayer()
        if layer.GetFeatureCount() == 0:
            raise ValueError("No features found in vector file: %s", vector_file)

    except Exception as e:
        logger.error("Error reading vector file %s: %s", vector_file, e)
        raise

    if raster.geotransform is None:
        raise ValueError("Raster must have valid geotransform information")

    rows, cols = raster.data.shape[:2]

    driver = gdal.GetDriverByName('MEM')
    mem_raster = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(raster.geotransform)
    layer_srs = layer.GetSpatialRef()
    if layer_srs:
        mem_raster.SetProjection(layer_srs.ExportToWkt())

    gdal.RasterizeLayer(mem_raster, [1], layer, burn_values=[1])

    band = mem_raster.GetRasterBand(1)
    mask_array = band.ReadAsArray()

    mask = mask_array.astype(bool)

    mem_raster = None
    vector_ds = None

    logger.info("Mask created successfully. Valid pixels: %d/%d (%.1f%%)",
                np.sum(mask), mask.size, 100.0 * np.sum(mask) / mask.size)

    return mask
