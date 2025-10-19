import logging
from typing import List
import numpy as np
from ..geo_reader import Raster, create_mask_location

logger = logging.getLogger(__name__)

class Standardizer:
    """
    Standardization preprocessor for handling nc and tif format climate data
    """

    def __init__(self, dimension: int, resolution: float = 0.5):
        """
        Initialize standardizer

        Args:
            dimension: Number of variables
            resolution: Resolution in degrees
        """
        self.dimension = dimension
        self.mean = np.zeros(dimension)
        self.std = np.zeros(dimension)
        self.is_fitted = False
        self.resolution = resolution

    def fit(self, input_data: Raster) -> None:
        """
        Calculate statistics and return standardized data for input array

        Args:
            input_data: Input data raster object

        Returns:
            Standardized data array with same shape as input
        """
        # Check input dimensions
        if input_data.data.shape[0] != self.dimension:
            raise ValueError(f"Input data length ({len(input_data)}) does not match expected dimension ({self.dimension})")

        logger.info("Computing statistics...")
        data_matrix = input_data.data # shape: (dim, x, y)
        reshaped_data = data_matrix.reshape(self.dimension, -1)  # (dim, x*y)
        self.mean = np.nanmean(reshaped_data, axis=1)  # (dim,)
        self.std = np.nanstd(reshaped_data, axis=1)    # (dim,)

        # Avoid division by zero
        self.std = np.where(self.std == 0, 1e-8, self.std)
        self.is_fitted = True

        logger.info("Statistics computed:")
        logger.info("Mean range: [%.4f, %.4f]", self.mean.min(), self.mean.max())
        logger.info("Std range: [%.4f, %.4f]", self.std.min(), self.std.max())

    def standardize(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Apply z-score standardization to new sample data using fitted statistics

        Args:
            raw_data: numpy array with shape (dim, x, y) or (dim, num)

        Returns:
            standardized data with shape (num, dim)
        """
        if not self.is_fitted:
            raise ValueError("Please call fit() method first to compute statistics")

        if raw_data.shape[0] != self.dimension:
            raise ValueError(f"Input data length ({len(raw_data)}) does not match expected dimension ({self.dimension})")

        if len(raw_data.shape) == 2:
            mean_reshaped = self.mean.reshape(-1, 1)  # (dim, 1)
            std_reshaped = self.std.reshape(-1, 1)   # (dim, 1)
            standardized_data = (raw_data - mean_reshaped) / std_reshaped
        elif len(raw_data.shape) == 3:
            mean_reshaped = self.mean.reshape(-1, 1, 1)  # (dim, 1, 1)
            std_reshaped = self.std.reshape(-1, 1, 1)   # (dim, 1, 1)
            standardized_data = ((raw_data - mean_reshaped) / std_reshaped).reshape(self.dimension, -1)
        else:
            raise ValueError(f"Input data shape ({raw_data.shape}) is not expected (dim, x, y), please check the input data")

        return standardized_data.T

    def _create_land_mask(self, reference_raster: Raster) -> None:
        """
        [Deprecated] Create land mask from vector file using reference raster

        Args:
            reference_raster: Reference raster object for spatial parameters

        Returns:
            Boolean mask array, True indicates land
        """
        pass
        #try:
        #    mask_data = create_mask_location(
        #        vector_file=self.land_mask_path,
        #        raster=reference_raster
        #    )
        #    self.land_mask_array = mask_data
        #except Exception as e:
        #    logger.error("Error creating land mask from vector file: %s", e)
        #    raise

    def _apply_land_mask(self, data: Raster) -> np.ndarray:
        """
        [Deprecated] Apply land mask to extract land points from raster data

        Args:
            data: Input raster object

        Returns:
            Land points data as numpy array with shape (valid_locations, bands)
        """
        #if self.land_mask_array is None:
        #    self._create_land_mask(data)

        ## Ensure correct indexing: reshape 3D data to 2D, apply 1D mask, keep band dimension
        #height, width, bands = data.data.shape
        #reshaped_data = data.data.reshape(height * width, bands)  # (H*W, bands)
        #flat_mask = self.land_mask_array.flatten()  # (H*W,)

        #return reshaped_data[flat_mask]  # (valid_locations, bands)

    def _reshape_raster_data(self, input_data: List[Raster]) -> np.ndarray:
        """
        [Deprecated] Reshape to one dimension by stacking locations

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
            raise ValueError(f"Expected raster shape (height, width, {self.dimension}), but got {first_raster.shape}")

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
