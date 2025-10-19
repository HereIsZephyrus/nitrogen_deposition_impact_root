"""
Sample data reader for CSV files containing field measurements
"""
import os
import logging
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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

    def __init__(self, sample_file: str, grouped: bool = True):
        """
        Initialize sample reader

        Args:
            sample_file: Path to CSV file
        """
        self.sample_file = sample_file
        self._data: Optional[pd.DataFrame] = None
        self.grouped = grouped
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
        Get sample locations as (longitude, latitude) tuples, deduplicated by group
        (only the first location for each group ID is returned)

        Returns:
            List of (longitude, latitude) coordinate tuples
        """
        if self._data is None:
            return []

        locations = []
        seen_groups = set()

        for _, row in self._data.iterrows():
            try:
                # Check if group column exists and get group id
                if 'group' in self._data.columns and self.grouped:
                    group_id = int(row['group'])
                    if group_id in seen_groups:
                        continue  # Skip this row if we've already seen this group
                    seen_groups.add(group_id)

                lon = float(row['longitude'])
                lat = float(row['latitude'])
                if not np.isnan(lon) and not np.isnan(lat):
                    locations.append((lon, lat))
                else:
                    logger.warning("Invalid coordinates in row %s", row.get('data_id', 'unknown'))
            except (ValueError, KeyError) as e:
                logger.warning("Error parsing coordinates or group: %s", e)
                continue

        logger.info("Extracted %d unique group locations (original: %d records)", 
                   len(locations), len(self._data))
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

        # Deduplicate by group if group column exists
        if 'group' in self._data.columns and self.grouped:
            # Keep first occurrence of each group
            deduplicated_data = self._data.drop_duplicates(subset=['group'], keep='first')
        else:
            deduplicated_data = self._data

        # Use pd.to_numeric to safely convert to float
        soil_data = []
        for col in available_cols:
            numeric_values = pd.to_numeric(deduplicated_data[col], errors='coerce').values
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

        # Deduplicate by group if group column exists
        if 'group' in self._data.columns and self.grouped:
            deduplicated_data = self._data.drop_duplicates(subset=['group'], keep='first')
        else:
            deduplicated_data = self._data

        # Process each column
        nitrogen_data = []

        for col in self.NITROGEN_COLUMNS:
            if col not in deduplicated_data.columns:
                logger.warning("Column %s not found, skipping", col)
                continue

            if col in ['n_addition', 'duration']:
                # Numeric columns - use pd.to_numeric with errors='coerce' to handle non-numeric values
                numeric_values = pd.to_numeric(deduplicated_data[col], errors='coerce').values
                nitrogen_data.append(numeric_values)
            else:
                # Categorical columns (fertilizer_type, treatment_date)
                cat_data = deduplicated_data[col].astype('category')
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

        # Deduplicate by group if group column exists
        if 'group' in self._data.columns and self.grouped:
            deduplicated_data = self._data.drop_duplicates(subset=['group'], keep='first')
        else:
            deduplicated_data = self._data

        # Encode vegetation types as categories
        vegetation = deduplicated_data[col_name].astype('category')
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

    def get_categorical(self) -> np.ndarray:
        """
        Get all categorical variables (fertilizer_type, treatment_date, vegetation_type)

        Returns:
            Array with shape (n_samples, n_categorical_features) containing category codes
        """
        if self._data is None:
            return np.array([])

        # Deduplicate by group if group column exists
        if 'group' in self._data.columns and self.grouped:
            deduplicated_data = self._data.drop_duplicates(subset=['group'], keep='first')
        else:
            deduplicated_data = self._data

        categorical_data = []

        # Get vegetation type
        if 'ecosystem/vegetation_type' in deduplicated_data.columns:
            veg_data = deduplicated_data['ecosystem/vegetation_type'].astype('category')
            categorical_data.append(veg_data.cat.codes.values.astype(float))

        # Stack columns if we have any categorical data
        if categorical_data:
            result = np.column_stack(categorical_data)
            logger.info("Extracted categorical data: %d samples, %d features", 
                       result.shape[0], result.shape[1])
        else:
            logger.warning("No categorical columns found in dataset")
            result = np.array([])

        return result

    def get_categorical_onehot(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get all categorical variables as one-hot encoded arrays

        Returns:
            Tuple of (one_hot_encoded_array, feature_names)
            - one_hot_encoded_array: Array with shape (n_samples, n_onehot_features)
            - feature_names: List of feature names for each one-hot column
        """
        if self._data is None:
            return np.array([]), []

        # Deduplicate by group if group column exists
        if 'group' in self._data.columns and self.grouped:
            deduplicated_data = self._data.drop_duplicates(subset=['group'], keep='first')
        else:
            deduplicated_data = self._data

        # Collect all categorical columns and their data
        categorical_columns = []
        categorical_arrays = []

        # Get fertilizer type
        if 'fertilizer_type' in deduplicated_data.columns:
            fert_data = deduplicated_data['fertilizer_type'].values.reshape(-1, 1)
            categorical_columns.append('fertilizer_type')
            categorical_arrays.append(fert_data)

        # Get treatment date
        if 'treatment_date' in deduplicated_data.columns:
            treat_data = deduplicated_data['treatment_date'].values.reshape(-1, 1)
            categorical_columns.append('treatment_date')
            categorical_arrays.append(treat_data)

        # Get vegetation type
        if 'ecosystem/vegetation_type' in deduplicated_data.columns:
            veg_data = deduplicated_data['ecosystem/vegetation_type'].values.reshape(-1, 1)
            categorical_columns.append('ecosystem/vegetation_type')
            categorical_arrays.append(veg_data)

        if not categorical_arrays:
            logger.warning("No categorical columns found for one-hot encoding")
            return np.array([]), []

        # Combine all categorical data
        all_categorical_data = np.concatenate(categorical_arrays, axis=1)

        # Apply one-hot encoding
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        onehot_encoded = encoder.fit_transform(all_categorical_data)

        # Generate feature names
        feature_names = []
        for i, col_name in enumerate(categorical_columns):
            categories = encoder.categories_[i]
            for category in categories:
                feature_names.append(f"{col_name}_{category}")

        logger.info("One-hot encoded categorical data: %d samples, %d features", 
                   onehot_encoded.shape[0], onehot_encoded.shape[1])

        return onehot_encoded, feature_names

    def get_categorical_info(self) -> Dict[str, List[str]]:
        """
        Get information about categorical variable categories

        Returns:
            Dictionary mapping column names to their unique categories
        """
        if self._data is None:
            return {}

        # Deduplicate by group if group column exists
        if 'group' in self._data.columns and self.grouped:
            deduplicated_data = self._data.drop_duplicates(subset=['group'], keep='first')
        else:
            deduplicated_data = self._data

        categorical_info = {}

        # Get fertilizer type categories
        if 'fertilizer_type' in deduplicated_data.columns:
            categories = deduplicated_data['fertilizer_type'].dropna().unique().tolist()
            categorical_info['fertilizer_type'] = sorted(categories)

        # Get treatment date categories
        if 'treatment_date' in deduplicated_data.columns:
            categories = deduplicated_data['treatment_date'].dropna().unique().tolist()
            categorical_info['treatment_date'] = sorted(categories)

        # Get vegetation type categories
        if 'ecosystem/vegetation_type' in deduplicated_data.columns:
            categories = deduplicated_data['ecosystem/vegetation_type'].dropna().unique().tolist()
            categorical_info['ecosystem/vegetation_type'] = sorted(categories)

        logger.info("Categorical info extracted for %d variables", len(categorical_info))
        for col, cats in categorical_info.items():
            logger.info("  %s: %d categories (%s)", col, len(cats), ', '.join(map(str, cats[:5])) + ('...' if len(cats) > 5 else ''))

        return categorical_info

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
