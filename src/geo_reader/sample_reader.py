"""
Sample data reader for CSV files containing field measurements
"""
import os
import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class Sample:
    """
    Reader for sample data CSV files

    Reads field measurement data including location, soil properties,
    nitrogen addition, vegetation type, and climate data.
    """

    # Column name mappings (standardized)
    SOIL_COLUMNS = [
        'available_water_capacity_for_rootable_soil_depth',
        'coarse_fragments', 'sand', 'silt', 'clay', 'bulk',
        'organic_carbon_content', 'ph_in_water', 'total_nitrogen_content',
        'cn_ratio', 'cec_soil', 'cec_clay', 'cec_eff', 'teb', 'bsat',
        'alum_sat', 'esp', 'tcarbon_eq', 'gypsum', 'elec_cond'
    ]

    CLIMATE_COLUMNS = ['p', 'ta']

    NITROGEN_COLUMNS = ['n_addition', 'fertilizer_type', 'treatment_date', 'duration']

    def __init__(self, sample_file: str):
        """
        Initialize sample reader

        Args:
            sample_file: Path to CSV file
        """
        self.sample_file = sample_file
        self._data: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load CSV data into pandas DataFrame"""
        if not os.path.exists(self.sample_file):
            raise FileNotFoundError(f"Sample file not found: {self.sample_file}")

        try:
            self._data = pd.read_csv(self.sample_file)
            logger.info("Loaded sample data: %d records", len(self._data))
        except Exception as e:
            logger.error("Error loading sample file: %s", e)
            raise

    def get_location(self) -> List[Tuple[float, float]]:
        """
        Get sample locations as (longitude, latitude) tuples

        Returns:
            List of (longitude, latitude) coordinate tuples
        """
        if self._data is None:
            return []

        locations = []
        for _, row in self._data.iterrows():
            try:
                lon = float(row['longitude'])
                lat = float(row['latitude'])
                if not np.isnan(lon) and not np.isnan(lat):
                    locations.append((lon, lat))
                else:
                    logger.warning("Invalid coordinates or year in row %s", row.get('data_id', 'unknown'))
            except (ValueError, KeyError) as e:
                logger.warning("Error parsing coordinates or year: %s", e)
                continue

        logger.info("Extracted %d valid locations", len(locations))
        return locations

    def get_soil(self) -> np.ndarray:
        """
        Get soil property data

        Returns:
            Array with shape (n_samples, n_soil_properties)
        """
        if self._data is None:
            return np.array([])

        # Select available soil columns from the dataset
        available_cols = [col for col in self.SOIL_COLUMNS if col in self._data.columns]

        if not available_cols:
            logger.warning("No soil columns found in dataset")
            return np.array([])

        # Use pd.to_numeric to safely convert to float
        soil_data = []
        for col in available_cols:
            numeric_values = pd.to_numeric(self._data[col], errors='coerce').values
            soil_data.append(numeric_values)

        soil_data = np.column_stack(soil_data)
        logger.info("Extracted soil data: %d samples, %d properties",
                   soil_data.shape[0], soil_data.shape[1])

        return soil_data

    def get_nitrogen(self) -> np.ndarray:
        """
        Get nitrogen treatment data including addition rate, fertilizer type, treatment date, and duration

        Returns:
            Array with shape (n_samples, 4) containing:
            - n_addition: N addition rate (numeric)
            - fertilizer_type: Fertilizer type (encoded as category codes)
            - treatment_date: Treatment date (encoded as category codes)
            - duration: Duration in years (numeric)
        """
        if self._data is None:
            return np.array([])

        available_cols = [col for col in self.NITROGEN_COLUMNS if col in self._data.columns]

        if not available_cols:
            logger.warning("No nitrogen columns found in dataset")
            return np.array([])

        # Process each column
        nitrogen_data = []

        for col in self.NITROGEN_COLUMNS:
            if col not in self._data.columns:
                logger.warning("Column %s not found, skipping", col)
                continue

            if col in ['n_addition', 'duration']:
                # Numeric columns - use pd.to_numeric with errors='coerce' to handle non-numeric values
                numeric_values = pd.to_numeric(self._data[col], errors='coerce').values
                nitrogen_data.append(numeric_values)
            else:
                # Categorical columns (fertilizer_type, treatment_date)
                cat_data = self._data[col].astype('category')
                nitrogen_data.append(cat_data.cat.codes.values.astype(float))

        # Stack columns
        result = np.column_stack(nitrogen_data) if nitrogen_data else np.array([])

        logger.info("Extracted nitrogen data: %d samples, %d features",
                   result.shape[0] if len(result.shape) > 1 else len(result),
                   result.shape[1] if len(result.shape) > 1 else 1)

        return result

    def get_vegetation(self) -> np.ndarray:
        """
        Get vegetation type data (encoded as categorical)

        Returns:
            Array with shape (n_samples,) containing vegetation type codes
        """
        if self._data is None:
            return np.array([])

        col_name = 'ecosystem/vegetation_type'
        if col_name not in self._data.columns:
            logger.warning("Vegetation type column not found in dataset")
            return np.array([])

        # Encode vegetation types as categories
        vegetation = self._data[col_name].astype('category')
        vegetation_codes = vegetation.cat.codes.values

        logger.info("Extracted vegetation data: %d samples, %d unique types",
                   len(vegetation_codes), len(vegetation.cat.categories))

        return vegetation_codes

    def get_climate(self) -> np.ndarray:
        """
        Get climate data (precipitation and temperature)

        Returns:
            Array with shape (n_samples, n_climate_variables)
        """
        if self._data is None:
            return np.array([])

        available_cols = [col for col in self.CLIMATE_COLUMNS if col in self._data.columns]

        if not available_cols:
            logger.warning("No climate columns found in dataset")
            return np.array([])

        # Use pd.to_numeric to handle special characters like en-dash
        climate_data = []
        for col in available_cols:
            numeric_values = pd.to_numeric(self._data[col], errors='coerce').values
            climate_data.append(numeric_values)

        climate_data = np.column_stack(climate_data)
        logger.info("Extracted climate data: %d samples, %d variables",
                   climate_data.shape[0], climate_data.shape[1])

        return climate_data

    def get_group(self) -> List[int]:
        """
        Get group labels

        Returns:
            List of group IDs
        """
        if self._data is None:
            return []

        if 'group' not in self._data.columns:
            logger.warning("Group column not found in dataset")
            return []

        groups = self._data['group'].values.astype(int).tolist()
        logger.info("Extracted group data: %d samples", len(groups))

        return groups

    def get_biomass(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get root biomass data (N addition and control)

        Returns:
            Tuple of (N_addition_biomass, control_biomass) arrays
        """
        if self._data is None:
            return np.array([]), np.array([])

        n_col = 'root_biomass_n_addation'
        ck_col = 'root_biomass_ck'

        n_biomass = self._data[n_col].values.astype(float) if n_col in self._data.columns else np.array([])
        ck_biomass = self._data[ck_col].values.astype(float) if ck_col in self._data.columns else np.array([])

        logger.info("Extracted biomass data: %d samples", len(n_biomass))

        return n_biomass, ck_biomass

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the raw pandas DataFrame

        Returns:
            Complete DataFrame with all columns
        """
        return self._data.copy() if self._data is not None else pd.DataFrame()

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self._data) if self._data is not None else 0

    def __repr__(self) -> str:
        """String representation"""
        if self._data is None:
            return "SampleReader(no data loaded)"
        return f"SampleReader({len(self._data)} samples, {len(self._data.columns)} columns)"
