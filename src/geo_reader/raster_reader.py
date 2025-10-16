"""
Raster data reader supporting NetCDF, HDF, and GeoTIFF formats
"""
import logging
from pathlib import Path
import numpy as np
import xarray as xr
from osgeo import gdal, osr

gdal.UseExceptions()

logger = logging.getLogger(__name__)

class Raster:
    """
    Raster data class
    """
    def __init__(self, data: np.ndarray, geotransform: tuple, resolution: float = 0.5):
        """
        Initialize raster data

        Args:
            data: Data array
            geotransform: Geotransform parameters
            resolution: Resolution in degrees
        """
        self.data = data
        self.geotransform = geotransform
        self.resolution = resolution

    def update_data(self, data: np.ndarray) -> None:
        """
        Update raster data

        Args:
            data: Data array
        """
        self.data = data

class RasterReader:
    """
    Raster data reader supporting NetCDF, HDF, and GeoTIFF formats
    """

    @staticmethod
    def read_file(file_path: str, resolution: float = 0.5) -> Raster:
        """
        Read raster file (nc/nc4/hdf/hdf5/tif/tiff)

        Args:
            file_path: Path to the raster file
            resolution: Resolution in degrees
        Returns:
            Raster object
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() in ['.nc', '.nc4', '.hdf', '.hdf5']:
            logger.info("Reading NetCDF file: %s", file_path)
            try:
                raster = RasterReader._read_netcdf(file_path, resolution)
            except Exception as e:
                logger.error("Error reading NetCDF file: %s", e)
                raise e
            logger.info("NetCDF file read successfully: %s", file_path)
            return raster
        elif file_path.suffix.lower() in ['.tif', '.tiff']:
            logger.info("Reading GeoTIFF file: %s", file_path)
            try:
                raster = RasterReader._read_geotiff(file_path, resolution)
            except Exception as e:
                logger.error("Error reading GeoTIFF file: %s", e)
                raise e
            logger.info("GeoTIFF file read successfully: %s", file_path)
            return raster
        else:
            logger.error("Unsupported file format: %s", file_path.suffix)
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    @staticmethod
    def _read_netcdf(file_path: Path, resolution: float = 0.5) -> Raster:
        """
        Read NetCDF file

        Args:
            file_path: Path to NetCDF file
            resolution: Resolution in degrees
        Returns:
            Raster object
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
                    geotransform = (lon.min() - resolution/2, resolution, 0,
                                  lat.max() + resolution/2, 0, -resolution)

            return Raster(data, geotransform, resolution)

    @staticmethod
    def _read_geotiff(file_path: Path, resolution: float = 0.5) -> Raster:
        """
        Read GeoTIFF file using GDAL

        Args:
            file_path: Path to GeoTIFF file
            resolution: Resolution in degrees
        Returns:
            Raster object
        """
        dataset = gdal.Open(str(file_path))
        if dataset is None:
            raise ValueError(f"Cannot open TIFF file: {file_path}")

        # Get geotransform parameters
        geotransform = dataset.GetGeoTransform()

        # Read data
        if dataset.RasterCount == 1:
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
        else:
            data = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount))
            for i in range(dataset.RasterCount):
                band = dataset.GetRasterBand(i + 1)
                data[:, :, i] = band.ReadAsArray()

        dataset = None  # Close dataset
        return Raster(data, geotransform, resolution)

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
