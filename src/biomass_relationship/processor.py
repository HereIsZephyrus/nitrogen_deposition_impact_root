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
    # Apply z-score standardization to each column
    raw_data = (raw_data - raw_data.mean()) / raw_data.std()
    climate_group = get_climate_group(os.path.join(statistic_folder_path, 'sample_cluster_result.csv'))

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

    # exclude dropped rows
    climate_group = climate_group[raw_data.index]
    total_reference_group = np.array(total_reference_group)[total_data.index]
    total_climate_group = total_climate_group[total_data.index]
    biomass_add, biomass_ck = total_sample.get_biomass()
    nitrogen_add = total_sample.get_nitrogen()
    biomass_add = biomass_add[total_data.index]
    biomass_ck = biomass_ck[total_data.index]
    nitrogen_add = nitrogen_add[total_data.index]
    total_categorical_data = total_categorical_data[total_data.index]

    n_to_soil_influencer = NitrogenToSoilInfluencer(
        nitrogen_impact_data_path=soil_impact_data_path,
        time_impact_data_path=time_impact_data_path
    )

    for group in np.unique(climate_group):
        group_nitrogen_add = nitrogen_add[total_climate_group == group]
        if (len(group_nitrogen_add) < 3):
            logger.warning(f"Group {group} has less than 5 samples, skipping")
            continue
        group_biomass_add = np.log(biomass_add[total_climate_group == group])
        group_biomass_ck = np.log(biomass_ck[total_climate_group == group])
        group_average = np.mean(
            np.stack([group_biomass_add, group_biomass_ck]),
            axis=0
        )
        group_average = (group_average - np.mean(group_average)) / np.std(group_average) # standardize average biomass
        group_model_data = total_data[total_climate_group == group]
        group_categorical_data = total_categorical_data[total_climate_group == group]
        transformed_data_ck = pca_analyzer.transform(group_model_data)
        input_ck = np.concatenate([transformed_data_ck, group_categorical_data, group_average.reshape(-1,1)], axis=1)
        impacted_data = n_to_soil_influencer.impact(
            group_model_data,
            group_nitrogen_add[:, 0], # addition rate
            group_nitrogen_add[:, 2], # duration
        )
        transformed_data_add = pca_analyzer.transform(impacted_data)
        input_add = np.concatenate([transformed_data_add, group_categorical_data, group_average.reshape(-1,1)], axis=1)
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
        trainer.train_nls(
            n_addtion = group_n_addtion,
            alpha_l1 = 0.1,
            alpha_l2 = 0.02,
            auto_regularization = True,
            save_to_file = True,
            plot = True
        )
        trainer.train_svm(
            n_addtion = group_n_addtion,
            auto_tune = True,
            intensive_search = True,
            save_to_file = True,
            plot = True
        )
        trainer.train_decision_tree(
            n_addtion = group_n_addtion, 
            max_depth = 10, 
            min_samples_split = 2, 
            min_samples_leaf = 1, 
            learning_rate = 0.1,
            n_estimators = 100,
            save_to_file = True, 
            plot = True
        )

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
