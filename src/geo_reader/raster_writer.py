import os
import logging
import numpy as np
from osgeo import gdal
from .raster_reader import Raster

logger = logging.getLogger(__name__)

def write_raster(data: np.ndarray, ref_raster: Raster, output_path: str) -> None:
    """
    Write raster to file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # reshape data from (label,) to (x,y)
    reshaped_data = data.reshape(ref_raster.data.shape[1], ref_raster.data.shape[2])
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, reshaped_data.shape[1], reshaped_data.shape[0], 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(ref_raster.geotransform)
    dataset.SetProjection('EPSG:4326')
    dataset.WriteArray(reshaped_data)
    dataset.FlushCache()
    dataset.Close()
    dataset = None
    driver = None
    logger.info(f"Raster written to {output_path}")
