import logging
from typing import List, Union
import numpy as np
from ..geo_reader import Raster, create_mask_location

logger = logging.getLogger(__name__)

class Standardizer:
    """
    Standardization preprocessor for handling nc and tif format climate data
    """

    def __init__(self, dimension: int, land_mask_path: str, resolution: float = 0.5):
        """
        Initialize standardizer

        Args:
            dimension: Number of variables
            land_mask_path: Land mask vector file path (shp or gpkg format)
            resolution: Resolution in degrees
        """
        self.dimension = dimension
        self.land_mask_path = land_mask_path
        self.land_mask_array = None
        self.mean = np.zeros(dimension)
        self.std = np.zeros(dimension)
        self.is_fitted = False
        self.resolution = resolution

    def fit(self, input_data: List[Raster]) -> None:
        """
        Calculate statistics and return standardized data for input array

        Args:
            input_data: Input data list of raster objects

        Returns:
            Standardized data array with same shape as input
        """
        # Check input dimensions
        if len(input_data) != self.dimension:
            raise ValueError(f"Input data length ({len(input_data)}) does not match expected dimension ({self.dimension})")

        logger.info("Computing statistics...")
        data_matrix = self._reshape_raster_data(input_data) # reshape to (location, dimension)
        self.mean = np.mean(data_matrix, axis=0)
        self.std = np.std(data_matrix, axis=0)

        # Avoid division by zero
        self.std = np.where(self.std == 0, 1e-8, self.std)
        self.is_fitted = True

        logger.info("Statistics computed:")
        logger.info("Mean range: [%.4f, %.4f]", self.mean.min(), self.mean.max())
        logger.info("Std range: [%.4f, %.4f]", self.std.min(), self.std.max())

    def standardize(self, raw_data: Union[Raster, List[Raster], np.ndarray]) -> np.ndarray:
        """
        Apply z-score standardization to new sample data using fitted statistics

        Args:
            raw_data: Raster object or List[Raster] or numpy array
        """
        if not self.is_fitted:
            raise ValueError("Please call fit() method first to compute statistics")

        data = None
        if isinstance(raw_data, Raster):
            data = self._apply_land_mask(raw_data)
        elif isinstance(raw_data, list):
            data = self._reshape_raster_data(raw_data)
        elif isinstance(raw_data, np.ndarray):
            data = raw_data

        try:
            standardized_data = (data - self.mean) / self.std
        except Exception as e:
            logger.error("Error standardizing data: %s", e)
            raise

        logger.info("Standardization complete, output shape: %s", standardized_data.shape)
        return standardized_data

    def _create_land_mask(self, reference_raster: Raster) -> None:
        """
        Create land mask from vector file using reference raster

        Args:
            reference_raster: Reference raster object for spatial parameters

        Returns:
            Boolean mask array, True indicates land
        """
        try:
            mask_data = create_mask_location(
                vector_file=self.land_mask_path,
                raster=reference_raster
            )
            self.land_mask_array = mask_data
        except Exception as e:
            logger.error("Error creating land mask from vector file: %s", e)
            raise

    def _apply_land_mask(self, data: Raster) -> np.ndarray:
        """
        Apply land mask to extract land points from raster data

        Args:
            data: Input raster object

        Returns:
            Land points data as numpy array with shape (valid_locations, bands)
        """
        if self.land_mask_array is None:
            self._create_land_mask(data)

        # Ensure correct indexing: reshape 3D data to 2D, apply 1D mask, keep band dimension
        height, width, bands = data.data.shape
        reshaped_data = data.data.reshape(height * width, bands)  # (H*W, bands)
        flat_mask = self.land_mask_array.flatten()  # (H*W,)

        return reshaped_data[flat_mask]  # (valid_locations, bands)

    def _reshape_raster_data(self, input_data: List[Raster]) -> np.ndarray:
        """
        Reshape to one dimension by stacking locations

        Args:
            input_data: List of raster objects, each with shape (height, width, bands)

        Returns:
            Reshaped data matrix (total_locations, bands)
        """
        if not input_data:
            raise ValueError("Input data list cannot be empty")

        # Get band count from first raster
        first_raster = input_data[0].data
        if len(first_raster.shape) != 3 or first_raster.shape[2] != self.dimension:
            raise ValueError("Expected raster shape (height, width, %s), but got %s", self.dimension, first_raster.shape)

        # Collect all locations from all rasters
        all_locations = []

        for raster in input_data:
            data = self._apply_land_mask(raster)
            all_locations.append(data)

        result = np.concatenate(all_locations, axis=0)
        logger.info("Output matrix shape: %s (total_locations, bands)", result.shape)
        return result

    def __repr__(self) -> str:
        header = f"Standardizer(dimension={self.dimension}, resolution={self.resolution})"
        mean_info = f"mean: {self.mean}"
        std_info = f"std: {self.std}"
        return f"{header}\n{'-'*len(header)}\n{mean_info}\n{'-'*len(mean_info)}\n{std_info}\n{'-'*len(std_info)}"
