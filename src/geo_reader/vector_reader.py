import logging
import numpy as np
from osgeo import gdal, ogr
from pathlib import Path

gdal.UseExceptions()

logger = logging.getLogger(__name__)

class VectorReader:
    """
    Vector data reader supporting Shapefile and GeoPackage formats
    """

    @staticmethod
    def create_raster_mask(vector_path: str, 
                          reference_geotransform: tuple, 
                          reference_shape: tuple) -> np.ndarray:
        """
        Create raster mask from vector file

        Args:
            vector_path: Path to vector file (shp or gpkg)
            reference_geotransform: Reference raster geotransform parameters
            reference_shape: Reference raster shape (height, width)

        Returns:
            Boolean mask array, True indicates areas inside vector polygons
        """
        vector_path = Path(vector_path)
        if not vector_path.exists():
            logger.error(f"Vector file does not exist: {vector_path}")
            raise FileNotFoundError(f"Vector file does not exist: {vector_path}")

        # Open vector file
        logger.info(f"Opening vector file: {vector_path}")
        vector_ds = VectorReader._open_vector_file(vector_path)
        layer = vector_ds.GetLayer()

        # Create target raster dataset (in memory)
        height, width = reference_shape[:2]
        mem_ds = VectorReader._create_memory_raster(width, height, reference_geotransform, layer)

        # Rasterize vector
        band = mem_ds.GetRasterBand(1)
        band.Fill(0)  # Background value is 0

        # Execute rasterization, vector interior area value is 1
        gdal.RasterizeLayer(mem_ds, [1], layer, burn_values=[1])

        # Read result
        mask_array = band.ReadAsArray()

        # Cleanup
        mem_ds = None
        vector_ds = None

        return mask_array.astype(bool)

    @staticmethod
    def _open_vector_file(vector_path: Path):
        """
        Open vector file using appropriate OGR driver

        Args:
            vector_path: Path to vector file

        Returns:
            OGR DataSource object
        """
        if vector_path.suffix.lower() == '.shp':
            driver = ogr.GetDriverByName('ESRI Shapefile')
        elif vector_path.suffix.lower() == '.gpkg':
            driver = ogr.GetDriverByName('GPKG')
        else:
            logger.error(f"Unsupported vector format: {vector_path.suffix}")
            raise ValueError(f"Unsupported vector format: {vector_path.suffix}")

        logger.info(f"Opening vector file: {vector_path}")
        vector_ds = driver.Open(str(vector_path), 0)
        if vector_ds is None:
            raise ValueError(f"Cannot open vector file: {vector_path}")

        return vector_ds

    @staticmethod
    def _create_memory_raster(width: int, height: int, geotransform: tuple, layer):
        """
        Create memory raster dataset for rasterization

        Args:
            width: Raster width
            height: Raster height
            geotransform: Geotransform parameters
            layer: Vector layer

        Returns:
            GDAL memory dataset
        """
        mem_driver = gdal.GetDriverByName('MEM')
        mem_ds = mem_driver.Create('', width, height, 1, gdal.GDT_Byte)

        # Set geotransform and projection
        mem_ds.SetGeoTransform(geotransform)

        # Get vector spatial reference system
        layer_srs = layer.GetSpatialRef()
        if layer_srs is not None:
            mem_ds.SetProjection(layer_srs.ExportToWkt())

        return mem_ds

    @staticmethod
    def get_vector_info(vector_path: str) -> dict:
        """
        Get basic information about vector file

        Args:
            vector_path: Path to vector file

        Returns:
            Dictionary containing vector information
        """
        vector_path = Path(vector_path)
        vector_ds = VectorReader._open_vector_file(vector_path)
        layer = vector_ds.GetLayer()

        info = {
            'feature_count': layer.GetFeatureCount(),
            'geometry_type': layer.GetGeomType(),
            'spatial_ref': layer.GetSpatialRef(),
            'extent': layer.GetExtent()  # (min_x, max_x, min_y, max_y)
        }

        vector_ds = None
        return info
