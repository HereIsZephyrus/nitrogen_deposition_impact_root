import numpy as np
from typing import Union, List, Tuple
from ..geo_reader import RasterReader, VectorReader

class Standardizer:
    """
    Standardization preprocessor for handling nc and tif format climate data
    Supports multivariate (20 indicators) z-score standardization
    """

    def __init__(self, dimension: int, land_mask_path: str):
        """
        Initialize standardizer

        Args:
            dimension: Number of variable dimensions
            land_mask_path: Land mask vector file path (shp or gpkg format)
        """
        self.dimension = dimension
        self.land_mask_path = land_mask_path
        self.land_mask_array = None
        self.mean = np.zeros(dimension)
        self.std = np.zeros(dimension)
        self.is_fitted = False

    def _read_raster(self, file_path: str) -> np.ndarray:
        """
        Read raster data and return numpy array

        Args:
            file_path: Path to raster file

        Returns:
            Data as numpy array
        """
        data, _, _ = RasterReader.read_file(file_path)
        return data

    def _create_land_mask(self, reference_file: str) -> np.ndarray:
        """
        Create land mask from vector file using reference raster

        Args:
            reference_file: Path to reference raster file for spatial parameters

        Returns:
            Boolean mask array, True indicates land
        """
        # Get reference raster parameters
        _, geotransform, shape = RasterReader.read_file(reference_file)

        # Create mask using vector reader
        return VectorReader.create_raster_mask(
            self.land_mask_path, geotransform, shape
        )

    def _apply_land_mask(self, data: np.ndarray, reference_file: str = None) -> np.ndarray:
        """
        Apply land mask to extract land raster points

        Args:
            data: Input data as numpy array
            reference_file: Reference file path for generating mask (used only if mask not exist)

        Returns:
            1D array containing only land points
        """
        if self.land_mask_array is None:
            if reference_file is not None:
                # Generate and store land mask using reference file
                print("Generating land mask from vector file...")
                self.land_mask_array = self._create_land_mask(reference_file)
                land_pixels = np.sum(self.land_mask_array)
                total_pixels = self.land_mask_array.size
                print(f"Land mask generated: {land_pixels}/{total_pixels} land pixels ({land_pixels/total_pixels*100:.2f}%)")
            else:
                # Use valid (finite) data points as fallback
                valid_mask = RasterReader.get_valid_mask(data)
                return data[valid_mask]

        # Apply land mask
        return data[self.land_mask_array]

    def load_data_from_files(self, file_paths: List[str]) -> np.ndarray:
        """
        Load data from files and return numpy array suitable for fit method

        Args:
            file_paths: List of file paths containing variable data

        Returns:
            Data array with shape (dimension, N) where N is number of land points
        """
        if len(file_paths) != self.dimension:
            raise ValueError(f"Number of files ({len(file_paths)}) does not match dimension ({self.dimension})")

        all_data = []

        for i, file_path in enumerate(file_paths):
            print(f"Loading file {i+1}/{len(file_paths)}: {file_path}")

            # Read raster data
            data = self._read_raster(file_path)

            # Apply land mask to extract land points (generate mask on first call)
            reference_file = file_path if self.land_mask_array is None else None
            land_data = self._apply_land_mask(data, reference_file)

            all_data.append(land_data)

        # Convert to matrix form X^{dimension×N}
        # Ensure all variables have consistent number of land points
        min_length = min(len(arr) for arr in all_data)
        data_matrix = np.array([arr[:min_length] for arr in all_data])

        print(f"Data loaded, shape: {data_matrix.shape} (variables × land points)")
        return data_matrix

    def fit(self, input_data: np.ndarray) -> np.ndarray:
        """
        Calculate statistics and return standardized data for input array

        Args:
            input_data: Input data array with shape (dimension, N) where N is number of samples

        Returns:
            Standardized data array with same shape as input
        """
        # Check input dimensions
        if input_data.shape[0] != self.dimension:
            raise ValueError(f"Input data first dimension ({input_data.shape[0]}) does not match expected dimension ({self.dimension})")

        print("Computing statistics...")

        # Calculate mean and standard deviation for each variable
        self.mean = np.mean(input_data, axis=1)  # shape: (dimension,)
        self.std = np.std(input_data, axis=1)    # shape: (dimension,)

        # Avoid division by zero
        self.std = np.where(self.std == 0, 1e-8, self.std)

        self.is_fitted = True

        print("Statistics computed:")
        print(f"Mean range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
        print(f"Std range: [{self.std.min():.4f}, {self.std.max():.4f}]")

        # Apply z-score standardization to input data
        mean_broadcasted = self.mean.reshape(-1, 1)
        std_broadcasted = self.std.reshape(-1, 1)

        standardized_data = (input_data - mean_broadcasted) / std_broadcasted

        print(f"Standardization complete, output shape: {standardized_data.shape}")
        print(f"Standardized data range: [{standardized_data.min():.4f}, {standardized_data.max():.4f}]")

        return standardized_data

    def standardize(self, sample_data: Union[List[str], np.ndarray]) -> np.ndarray:
        """
        Apply z-score standardization to new sample data using fitted statistics

        Args:
            sample_data: New sample data, can be file path list or numpy array
                       - If file path list, length should match dimension
                       - If numpy array, shape should be (dimension, n) where n is number of samples

        Returns:
            Standardized data matrix Z^{dimension×n}, using formula z = (x - mean) / std
        """
        if not self.is_fitted:
            raise ValueError("Please call fit() method first to compute statistics")

        # Process input data
        if isinstance(sample_data, (list, tuple)):
            # Load sample data from files
            if len(sample_data) != self.dimension:
                raise ValueError(f"Number of files ({len(sample_data)}) does not match dimension ({self.dimension})")

            sample_matrix = []
            for i, file_path in enumerate(sample_data):
                data = self._read_raster(file_path)
                # Apply land mask to extract land points (generate mask on first call)
                reference_file = file_path if (self.land_mask_array is None and i == 0) else None
                land_data = self._apply_land_mask(data, reference_file)
                sample_matrix.append(land_data)

            # Ensure all variables have consistent number of sample points
            min_samples = min(len(arr) for arr in sample_matrix)
            X = np.array([arr[:min_samples] for arr in sample_matrix])

        elif isinstance(sample_data, np.ndarray):
            if sample_data.shape[0] != self.dimension:
                raise ValueError(f"Input data first dimension ({sample_data.shape[0]}) does not match dimension ({self.dimension})")
            X = sample_data
        else:
            raise ValueError("sample_data must be file path list or numpy array")

        # Apply z-score standardization: z = (x - mean) / std
        # Broadcast calculation, mean and std shape (dimension, 1) to multiply with X^{dimension×n}
        mean_broadcasted = self.mean.reshape(-1, 1)
        std_broadcasted = self.std.reshape(-1, 1)

        Z = (X - mean_broadcasted) / std_broadcasted

        print(f"Standardization complete, output shape: {Z.shape}")
        print(f"Standardized data range: [{Z.min():.4f}, {Z.max():.4f}]")

        return Z

    def get_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get computed statistical information

        Returns:
            (mean, std): Mean vector and standard deviation vector
        """
        if not self.is_fitted:
            raise ValueError("Please call fit() method first to compute statistics")

        return self.mean.copy(), self.std.copy()

    def save_statistics(self, file_path: str) -> None:
        """
        Save statistical information to file

        Args:
            file_path: Save path
        """
        if not self.is_fitted:
            raise ValueError("Please call fit() method first to compute statistics")

        np.savez(file_path, mean=self.mean, std=self.std)
        print(f"Statistics saved to: {file_path}")

    def load_statistics(self, file_path: str) -> None:
        """
        Load statistical information from file

        Args:
            file_path: File path
        """
        data = np.load(file_path)
        self.mean = data['mean']
        self.std = data['std']

        if len(self.mean) != self.dimension or len(self.std) != self.dimension:
            raise ValueError("Loaded statistics dimensions do not match current settings")

        self.is_fitted = True
        print(f"Statistics loaded from {file_path}")
