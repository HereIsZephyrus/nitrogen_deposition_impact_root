import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import xarray as xr
from osgeo import gdal

gdal.UseExceptions()

logger = logging.getLogger(__name__)

class RasterReader:
    """
    Raster data reader supporting NetCDF and GeoTIFF formats
    """

    @staticmethod
    def read_file(file_path: str) -> Tuple[np.ndarray, tuple, tuple]:
        """
        Read raster file (nc/nc4/tif/tiff)

        Args:
            file_path: Path to the raster file

        Returns:
            (data, geotransform, shape): Data array, geotransform parameters and shape
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() in ['.nc', '.nc4']:
            logger.info("Reading NetCDF file: %s", file_path)
            data, geotransform, shape = RasterReader._read_netcdf(file_path)
            logger.info("NetCDF file read successfully: %s", file_path)
            return data, geotransform, shape
        elif file_path.suffix.lower() in ['.tif', '.tiff']:
            logger.info("Reading GeoTIFF file: %s", file_path)
            data, geotransform, shape = RasterReader._read_geotiff(file_path)
            logger.info("GeoTIFF file read successfully: %s", file_path)
            return data, geotransform, shape
        else:
            logger.error("Unsupported file format: %s", file_path.suffix)
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    @staticmethod
    def _read_netcdf(file_path: Path) -> Tuple[np.ndarray, tuple, tuple]:
        """
        Read NetCDF file

        Args:
            file_path: Path to NetCDF file

        Returns:
            (data, geotransform, shape): Data array, geotransform parameters and shape
        """
        with xr.open_dataset(file_path) as ds:
            # Get first data variable
            data_vars = list(ds.data_vars.keys())
            if not data_vars:
                logger.error("No data variables found in NetCDF file %s", file_path)
                raise ValueError(f"No data variables found in NetCDF file {file_path}")

            data = ds[data_vars[0]].values

            # Try to get geographic coordinate information
            geotransform = None
            if 'lat' in ds.coords and 'lon' in ds.coords:
                lat = ds['lat'].values
                lon = ds['lon'].values
                if len(lat) > 1 and len(lon) > 1:
                    lat_res = abs(lat[1] - lat[0])
                    lon_res = abs(lon[1] - lon[0])
                    # Create simple geotransform (top_left_x, x_res, 0, top_left_y, 0, y_res)
                    geotransform = (lon.min() - lon_res/2, lon_res, 0, 
                                  lat.max() + lat_res/2, 0, -lat_res)

            return data, geotransform, data.shape

    @staticmethod
    def _read_geotiff(file_path: Path) -> Tuple[np.ndarray, tuple, tuple]:
        """
        Read GeoTIFF file using GDAL

        Args:
            file_path: Path to GeoTIFF file

        Returns:
            (data, geotransform, shape): Data array, geotransform parameters and shape
        """
        dataset = gdal.Open(str(file_path))
        if dataset is None:
            raise ValueError(f"Cannot open TIFF file: {file_path}")

        # Get geotransform parameters
        geotransform = dataset.GetGeoTransform()

        # Read data
        if dataset.RasterCount == 1:
            # Single band
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
        else:
            # Multi-band, read as (height, width, bands)
            data = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount))
            for i in range(dataset.RasterCount):
                band = dataset.GetRasterBand(i + 1)
                data[:, :, i] = band.ReadAsArray()

        shape = data.shape
        dataset = None  # Close dataset
        return data, geotransform, shape

    @staticmethod
    def get_valid_mask(data: np.ndarray) -> np.ndarray:
        """
        Get mask for valid (non-NaN, finite) data points

        Args:
            data: Input raster data

        Returns:
            Boolean mask array, True for valid points
        """
        if len(data.shape) == 2:
            return np.isfinite(data)
        else:
            # Multi-variable case, require all variables to be valid
            return np.all(np.isfinite(data), axis=2)
