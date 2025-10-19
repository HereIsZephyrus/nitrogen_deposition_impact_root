"""
Biomass relationship processor - integrates all analysis steps

This module implements the complete workflow for analyzing the relationship
between nitrogen deposition and vegetation root biomass, including:
1. Data preprocessing (standardization, one-hot encoding, N impact calculation)
2. Dimensionality reduction using PCA
3. Statistical modeling using multiple methods (NLS, XGBoost, LightGBM, SVM)
4. Feature importance extraction using SHAP
"""
import logging
import os
import pandas as pd
import numpy as np
from .train_model import ModelTrainer
from .pca import PCAnalyzer
from ..geo_reader import load_variance, Sample, get_climate_group
from ..n_impact_soil import NitrogenToSoilInfluencer
from ..variance import unpack_variance

logger = logging.getLogger(__name__)

def train(statistic_folder_path: str, climate_dir: str, output_dir: str):
    """
    Main entry point for biomass relationship analysis
    """
    sample_data_path = os.path.join(statistic_folder_path, 'sample.csv')
    soil_impact_data_path = os.path.join(statistic_folder_path, 'soil_change_nitrogen.csv')
    time_impact_data_path = os.path.join(statistic_folder_path, 'soil_change_year.csv')

    unique_sample = Sample(sample_data_path, grouped=True)
    input_var = load_variance(climate_dir=climate_dir, sample=unique_sample)

    raw_data = unpack_variance(input_var)
    raw_data = drop_nan(raw_data)
    climate_group = get_climate_group(os.path.join(output_dir, 'sample_cluster_result.csv'))

    pca_analyzer = PCAnalyzer(
        max_components=20,
        variance_threshold=0.95,
    )
    pca_analyzer.fit(raw_data)
    pca_analyzer.save(os.path.join(output_dir, 'pca_analyzer.csv'))

    total_sample = Sample(sample_data_path, grouped=False)
    total_categorical_data = total_sample.get_categorical()
    total_variance_data, total_reference_group = load_variance(climate_dir=climate_dir, sample=total_sample)
    total_data = unpack_variance(total_variance_data)
    total_data = drop_nan(total_data)
    total_climate_group = calculate_climate_group(total_reference_group, climate_group)
    # save total climate group to file(.csv)

    # exclude dropped rows
    climate_group = climate_group[raw_data.index]
    total_reference_group = np.array(total_reference_group)[total_data.index]
    total_climate_group = total_climate_group[total_data.index]
    total_categorical_data = total_categorical_data[total_data.index]

    # Get normalized biomass data by group
    biomass_add, biomass_ck = total_sample.get_biomass_normalized_by_group()
    nitrogen_add = total_sample.get_nitrogen()

    # Apply drop_nan indices to biomass and nitrogen data as well
    biomass_add = biomass_add[total_data.index]
    biomass_ck = biomass_ck[total_data.index]
    nitrogen_add = nitrogen_add[total_data.index]

    # Now exclude outliers from normalized data
    valid_index = exclude_outliers(np.concatenate([biomass_add, biomass_ck]))
    total_data = total_data.iloc[valid_index, :]
    biomass_add = biomass_add[valid_index]
    biomass_ck = biomass_ck[valid_index]
    nitrogen_add = nitrogen_add[valid_index]
    total_categorical_data = total_categorical_data[valid_index]
    total_reference_group = total_reference_group[valid_index]
    total_climate_group = total_climate_group[valid_index]

    n_to_soil_influencer = NitrogenToSoilInfluencer(
        nitrogen_impact_data_path=soil_impact_data_path,
        time_impact_data_path=time_impact_data_path
    )

    for group in np.unique(climate_group):
        group_biomass_add = biomass_add[total_climate_group == group]
        group_biomass_ck = biomass_ck[total_climate_group == group]
        group_nitrogen_add = nitrogen_add[total_climate_group == group]
        group_model_data = total_data[total_climate_group == group]
        group_categorical_data = total_categorical_data[total_climate_group == group]
        transformed_data_ck = pca_analyzer.transform(group_model_data)
        input_ck = np.concatenate([transformed_data_ck, group_categorical_data], axis=1)
        impacted_data = n_to_soil_influencer.impact(
            group_model_data,
            group_nitrogen_add[:, 0], # addition rate
            group_nitrogen_add[:, 2], # duration
        )
        transformed_data_add = pca_analyzer.transform(impacted_data)
        input_add = np.concatenate([transformed_data_add, group_categorical_data], axis=1)
        group_transformed_data = np.concatenate([input_ck, input_add], axis=0)
        group_dependence = np.concatenate([group_biomass_ck, group_biomass_add], axis=0)
        group_n_addtion = np.concatenate([
            np.zeros(len(group_nitrogen_add)),
            group_nitrogen_add[:, 0] * group_nitrogen_add[:, 2]
        ])
        trainer = ModelTrainer(
            X = group_transformed_data,
            y = group_dependence,
            group = group,
            output_dir = output_dir
        )
        trainer.train_nls(n_addtion = group_n_addtion, save_to_file = True, plot = True)
        #trainer.train_svm(n_addtion = group_n_addtion)
        #trainer.train_decision_tree(n_addtion = group_n_addtion)

def drop_nan(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Check and remove rows with NaN values from DataFrame

    Args:
        raw_data: Input DataFrame

    Returns:
        DataFrame with NaN rows removed

    Raises:
        ValueError: If no valid samples remain after removing NaN values
    """
    # Check for NaN values
    nan_counts = raw_data.isna().sum()
    if nan_counts.any():
        logger.warning("Found NaN values in data:")
        nan_cols = nan_counts[nan_counts > 0]
        for col, count in nan_cols.items():
            logger.warning("  Column '%s': %d NaN values (%.1f%%)", 
                          col, count, 100 * count / len(raw_data))

        # Remove rows with any NaN values
        n_before = len(raw_data)
        raw_data = raw_data.dropna()
        n_after = len(raw_data)
        logger.info("Dropped %d rows with NaN values (%d -> %d samples)", 
                   n_before - n_after, n_before, n_after)

        if n_after == 0:
            raise ValueError("No valid samples remaining after removing NaN values")
    else:
        logger.info("No NaN values found in data")

    return raw_data

def calculate_climate_group(reference_group: np.ndarray, unique_climate_group: np.ndarray) -> np.ndarray:
    """
    Re-encode reference group values to sequential integers starting from 0

    This function takes an array of group IDs and re-encodes them as sequential 
    integers (0, 1, 2, ...) based on the sorted order of unique values.

    Args:
        reference_group: Array of group IDs to be re-encoded
        unique_climate_group: Not used in this implementation (kept for compatibility)

    Returns:
        Array of sequential group IDs (0-based)
    """
    # Get unique values and sort them
    unique_values = np.unique(reference_group)

    value_to_index = {value: idx for idx, value in enumerate(unique_values)}
    reencoded_group = np.array([value_to_index[value] for value in reference_group])

    total_climate_group = np.zeros(len(reference_group))
    for idx, value in enumerate(reencoded_group):
        total_climate_group[idx] = unique_climate_group[value]

    return total_climate_group

def exclude_outliers(data: np.ndarray, z_threshold: float = 3.0) -> np.ndarray:
    """
    Exclude outliers from normalized data based on z-score threshold

    Args:
        data: Input array (should be normalized/standardized data)
        z_threshold: Z-score threshold for outlier detection (default: 3.0)

    Returns:
        Array of valid indices (non-outlier data points)
    """
    # First remove NaN values
    nan_mask = ~np.isnan(data)

    if not np.any(nan_mask):
        logger.warning("All data points are NaN")
        return np.array([], dtype=int)

    valid_data = data[nan_mask]

    # For normalized data, outliers are points with |z-score| > threshold
    # Since data should already be normalized, we can directly use the threshold
    outlier_mask = np.abs(valid_data) > z_threshold
    n_outliers = np.sum(outlier_mask)

    if n_outliers > 0:
        logger.info("Found %d outliers with |z-score| > %.1f (%.1f%% of data)", 
                   n_outliers, z_threshold, 100 * n_outliers / len(valid_data))
    else:
        logger.info("No outliers found with z-threshold %.1f", z_threshold)

    # Create final mask for original data array
    final_mask = np.zeros(len(data), dtype=bool)
    final_mask[nan_mask] = ~outlier_mask

    # Get indices of valid points
    valid_indices = np.where(final_mask)[0]

    logger.info("Valid data points after outlier removal: %d/%d (%.1f%%)",
               len(valid_indices), len(data), 100 * len(valid_indices) / len(data))

    # For concatenated arrays, split indices correctly
    half_length = len(data) // 2

    # Separate indices for first and second half of concatenated array
    first_half_indices = valid_indices[valid_indices < half_length]
    second_half_indices = valid_indices[valid_indices >= half_length] - half_length

    # Find common indices (samples valid in both arrays)
    common_indices = np.intersect1d(first_half_indices, second_half_indices)

    logger.info("Common valid indices after intersection: %d", len(common_indices))

    return common_indices
