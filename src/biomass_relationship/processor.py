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
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .pca import PCAnalyzer
from .svm import KernelSVMRegressor
from .shap import SHAPAnalyzer
from ..n_impact_soil.calculator import SoilCalculator, transfer_N_input

logger = logging.getLogger(__name__)


class BiomassRelationshipProcessor:
    """
    Main processor for biomass-nitrogen relationship analysis

    Integrates preprocessing, dimensionality reduction, statistical modeling,
    and feature importance analysis into a unified workflow.
    """

    def __init__(self,
                 sample_data_path: str,
                 soil_impact_data_path: str,
                 output_dir: str,
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.9,
                 max_components: int = 20,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the processor

        Args:
            sample_data_path: Path to sample CSV file
            soil_impact_data_path: Path to soil change data for N impact calculation
            output_dir: Directory for saving results
            n_components: Number of PCA components (None for automatic selection)
            variance_threshold: Variance threshold for PCA
            max_components: Maximum number of PCA components to retain
            test_size: Test set proportion
            random_state: Random seed for reproducibility
        """
        self.sample_data_path = sample_data_path
        self.soil_impact_data_path = soil_impact_data_path
        self.output_dir = output_dir
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.test_size = test_size
        self.random_state = random_state

        # Initialize components
        self.soil_calculator = None
        self.pca_analyzer = None
        self.scaler_continuous = StandardScaler()
        self.encoder_categorical = OneHotEncoder(drop='first', sparse_output=False)

        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Models
        self.models = {}
        self.model_results = {}
        self.shap_results = {}

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Initialized BiomassRelationshipProcessor")
        logger.info(f"Output directory: {self.output_dir}")

    def load_data(self) -> pd.DataFrame:
        """
        Load sample data from CSV file

        Returns:
            Loaded DataFrame
        """
        logger.info("Loading sample data from %s", self.sample_data_path)

        try:
            self.raw_data = pd.read_csv(self.sample_data_path)
            logger.info("Loaded %d samples", len(self.raw_data))
            logger.info("Columns: %s", list(self.raw_data.columns))

            return self.raw_data
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            raise

    def preprocess_data(self,
                       apply_nitrogen_impact: bool = True,
                       soil_method: str = 'linear') -> pd.DataFrame:
        """
        Preprocess data including standardization, encoding, and N impact calculation

        Step 1: Separate control (CK) and treatment (ADD) groups
        Step 2: Calculate atmospheric N deposition for control group
        Step 3: Apply N impact on soil properties for treatment group
        Step 4: Standardize continuous variables
        Step 5: One-hot encode categorical variables

        Args:
            apply_nitrogen_impact: Whether to apply nitrogen impact on soil properties
            soil_method: Method for soil impact calculation ('linear', 'spline', 'polynomial')

        Returns:
            Preprocessed DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting data preprocessing")
        logger.info("=" * 60)

        if self.raw_data is None:
            self.load_data()

        # Initialize soil impact calculator
        if apply_nitrogen_impact:
            logger.info("Initializing soil impact calculator with method: %s", soil_method)
            self.soil_calculator = SoilCalculator(
                data_path=self.soil_impact_data_path,
                method=soil_method
            )

        # Process each record - create both control and treatment samples
        processed_records = []

        for idx, row in self.raw_data.iterrows():
            # Extract basic information
            n_addition = row['n_addition']
            fertilizer_type = row['fertilizer_type']
            duration = row['duration']
            vegetation_type = row.get('ecosystem/vegetation_type', row.get('vegetation_type', 'unknown'))

            # Map vegetation type to numeric (0=forest, 1=grassland)
            if 'forest' in str(vegetation_type).lower():
                veg_code = 0
            elif 'grass' in str(vegetation_type).lower():
                veg_code = 1
            else:
                veg_code = 0  # Default to forest

            # Extract climate data
            climate_data = {
                'p': row['p'],
                'ta': row['ta']
            }

            # Extract soil properties
            soil_columns = [
                'available_water_capacity_for_rootable_soil_depth',
                'coarse_fragments', 'sand', 'silt', 'clay', 'bulk',
                'organic_carbon_content', 'ph_in_water', 'total_nitrogen_content',
                'cn_ratio', 'cec_soil', 'cec_clay', 'cec_eff', 'teb', 'bsat',
                'alum_sat', 'esp', 'tcarbon_eq', 'gypsum', 'elec_cond'
            ]

            soil_base = {col: row[col] for col in soil_columns if col in row.index}

            # Create control sample (CK group)
            # For control group, nitrogen is only from atmospheric deposition
            control_record = {
                'sample_id': f"{idx}_CK",
                'group': 'control',
                'root_biomass': row['root_biomass_ck'],
                'vegetation_type': veg_code,
                'n_deposition': 0,  # No additional N for control (atmospheric N already in soil)
                'fertilizer_type': 0,  # No fertilizer
                'treatment_date': row['treatment_date'],
                'duration': duration
            }
            control_record.update(climate_data)
            control_record.update(soil_base)
            processed_records.append(control_record)

            # Create treatment sample (ADD group)
            # For treatment group, calculate equivalent N deposition and adjust soil
            if apply_nitrogen_impact and self.soil_calculator is not None:
                # Convert fertilizer N to equivalent N deposition
                n_equivalent = transfer_N_input(n_addition, fertilizer_type)

                # Predict soil changes due to N addition
                try:
                    soil_changes = self.soil_calculator.predict_all(n_equivalent * duration)

                    # Apply changes to soil properties
                    soil_adjusted = soil_base.copy()
                    for prop, base_value in soil_base.items():
                        if prop in soil_changes and soil_changes[prop] is not None:
                            # Use the predicted value from soil calculator
                            soil_adjusted[prop] = soil_changes[prop]
                        else:
                            soil_adjusted[prop] = base_value

                except (ValueError, KeyError, RuntimeError) as e:
                    logger.warning("Failed to calculate soil impact for record %s: %s", idx, e)
                    soil_adjusted = soil_base.copy()
            else:
                soil_adjusted = soil_base.copy()

            # Map fertilizer type to numeric code
            fertilizer_map = {
                'NH4NO3': 1,
                '(NH4)2SO4': 2,
                'NaNO3': 3,
                'urea': 4
            }
            fertilizer_code = fertilizer_map.get(fertilizer_type, 0)

            treatment_record = {
                'sample_id': f"{idx}_ADD",
                'group': 'treatment',
                'root_biomass': row['root_biomass_n_addation'],
                'vegetation_type': veg_code,
                'n_deposition': n_addition,  # Additional N from fertilizer
                'fertilizer_type': fertilizer_code,
                'treatment_date': row['treatment_date'],
                'duration': duration
            }
            treatment_record.update(climate_data)
            treatment_record.update(soil_adjusted)
            processed_records.append(treatment_record)

        # Create DataFrame
        self.processed_data = pd.DataFrame(processed_records)

        logger.info("Preprocessing completed: %d samples created", len(self.processed_data))
        logger.info("  - Control samples: %d", len(self.processed_data[self.processed_data['group'] == 'control']))
        logger.info("  - Treatment samples: %d", len(self.processed_data[self.processed_data['group'] == 'treatment']))

        return self.processed_data

    def prepare_features(self,
                        apply_pca: bool = True,
                        save_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling

        Step 1: Separate continuous and categorical variables
        Step 2: Standardize continuous variables
        Step 3: One-hot encode categorical variables
        Step 4: Apply PCA for dimensionality reduction (optional)

        Args:
            apply_pca: Whether to apply PCA dimensionality reduction
            save_features: Whether to save feature data to file

        Returns:
            (features, target) tuple
        """
        logger.info("=" * 60)
        logger.info("Preparing features for modeling")
        logger.info("=" * 60)

        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        # Define feature columns
        continuous_features = [
            'p', 'ta',  # Climate
            'available_water_capacity_for_rootable_soil_depth',
            'coarse_fragments', 'sand', 'silt', 'clay', 'bulk',
            'organic_carbon_content', 'ph_in_water', 'total_nitrogen_content',
            'cn_ratio', 'cec_soil', 'cec_clay', 'cec_eff', 'teb', 'bsat',
            'alum_sat', 'esp', 'tcarbon_eq', 'gypsum', 'elec_cond',  # Soil
            'n_deposition', 'duration'  # Nitrogen
        ]

        categorical_features = [
            'vegetation_type', 'fertilizer_type'
        ]

        # Extract features
        X_continuous = self.processed_data[continuous_features].values
        X_categorical = self.processed_data[categorical_features].values
        y = self.processed_data['root_biomass'].values

        logger.info("Feature dimensions:")
        logger.info("  - Continuous features: %d", X_continuous.shape[1])
        logger.info("  - Categorical features: %d", X_categorical.shape[1])
        logger.info("  - Total samples: %d", len(y))

        # Standardize continuous features
        logger.info("Standardizing continuous features...")
        X_continuous_scaled = self.scaler_continuous.fit_transform(X_continuous)

        # One-hot encode categorical features
        logger.info("One-hot encoding categorical features...")
        X_categorical_encoded = self.encoder_categorical.fit_transform(X_categorical)

        logger.info("  - Encoded categorical dimension: %d", X_categorical_encoded.shape[1])

        # Combine features
        X_combined = np.hstack([X_continuous_scaled, X_categorical_encoded])

        # Create feature names
        categorical_feature_names = self.encoder_categorical.get_feature_names_out(categorical_features)
        all_feature_names = continuous_features + list(categorical_feature_names)

        logger.info("Combined feature dimension: %d", X_combined.shape[1])

        # Apply PCA if requested
        if apply_pca:
            logger.info("Applying PCA dimensionality reduction...")

            self.pca_analyzer = PCAnalyzer(
                n_components=self.n_components,
                variance_threshold=self.variance_threshold
            )

            X_pca = self.pca_analyzer.fit_transform(X_combined, feature_names=all_feature_names)

            # Limit to max_components
            if X_pca.shape[1] > self.max_components:
                logger.info("Limiting PCA components from %d to %d", X_pca.shape[1], self.max_components)
                X_pca = X_pca[:, :self.max_components]
                self.pca_analyzer.n_components = self.max_components

            _, cumulative_variance = self.pca_analyzer.get_explained_variance()
            logger.info("PCA results:")
            logger.info("  - Components: %d", X_pca.shape[1])
            logger.info("  - Variance explained: %.2f%%", cumulative_variance[min(X_pca.shape[1]-1, len(cumulative_variance)-1)] * 100)

            # Save PCA summary
            if save_features:
                pca_summary = self.pca_analyzer.get_summary()
                pca_summary_path = os.path.join(self.output_dir, 'pca_summary.csv')
                pca_summary.to_csv(pca_summary_path, index=False)
                logger.info("Saved PCA summary to %s", pca_summary_path)

            X_final = X_pca
            final_feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        else:
            X_final = X_combined
            final_feature_names = all_feature_names

        # Create DataFrame for features
        X_df = pd.DataFrame(X_final, columns=final_feature_names)
        y_series = pd.Series(y, name='root_biomass')

        # Save features if requested
        if save_features:
            features_path = os.path.join(self.output_dir, 'prepared_features.csv')
            full_data = pd.concat([X_df, y_series], axis=1)
            full_data.to_csv(features_path, index=False)
            logger.info("Saved prepared features to %s", features_path)

        logger.info("Feature preparation completed")

        return X_df, y_series

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into training and testing sets

        Args:
            X: Feature DataFrame
            y: Target Series
        """
        logger.info("=" * 60)
        logger.info("Splitting data into train/test sets")
        logger.info("=" * 60)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        logger.info("Split completed:")
        logger.info("  - Training samples: %d", len(self.X_train))
        logger.info("  - Testing samples: %d", len(self.X_test))

    def train_xgboost(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train XGBoost model

        Args:
            params: XGBoost parameters (optional)

        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("Training XGBoost model")
        logger.info("=" * 60)

        if self.X_train is None:
            raise ValueError("Data not split. Call split_data() first.")

        # Default parameters optimized for small sample size
        default_params = {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state
        }

        if params is not None:
            default_params.update(params)

        from xgboost import XGBRegressor
        model = XGBRegressor(**default_params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Metrics
        results = {
            'model': model,
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred
        }

        self.models['xgboost'] = model
        self.model_results['xgboost'] = results

        logger.info("XGBoost training completed:")
        logger.info("  - Train R²: %.4f", results['train_r2'])
        logger.info("  - Test R²: %.4f", results['test_r2'])
        logger.info("  - Train RMSE: %.4f", results['train_rmse'])
        logger.info("  - Test RMSE: %.4f", results['test_rmse'])

        return results

    def train_lightgbm(self, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train LightGBM model

        Args:
            params: LightGBM parameters (optional)

        Returns:
            Training results dictionary
        """
        logger.info("=" * 60)
        logger.info("Training LightGBM model")
        logger.info("=" * 60)

        if self.X_train is None:
            raise ValueError("Data not split. Call split_data() first.")

        # Default parameters optimized for small sample size
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'n_estimators': 100,
            'random_state': self.random_state,
            'verbose': -1
        }

        if params is not None:
            default_params.update(params)

        from lightgbm import LGBMRegressor
        model = LGBMRegressor(**default_params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        # Metrics
        results = {
            'model': model,
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred),
            'y_test_pred': y_test_pred,
            'y_train_pred': y_train_pred
        }

        self.models['lightgbm'] = model
        self.model_results['lightgbm'] = results

        logger.info("LightGBM training completed:")
        logger.info("  - Train R²: %.4f", results['train_r2'])
        logger.info("  - Test R²: %.4f", results['test_r2'])
        logger.info("  - Train RMSE: %.4f", results['train_rmse'])
        logger.info("  - Test RMSE: %.4f", results['test_rmse'])

        return results

    def train_svm(self,
                  kernels: Optional[List[str]] = None,
                  auto_tune: bool = True) -> Dict[str, Any]:
        """
        Train SVM models with different kernels

        Args:
            kernels: List of kernel types to try
            auto_tune: Whether to perform hyperparameter tuning

        Returns:
            Dictionary of results for each kernel
        """
        logger.info("=" * 60)
        logger.info("Training SVM models")
        logger.info("=" * 60)

        if self.X_train is None:
            raise ValueError("Data not split. Call split_data() first.")

        if kernels is None:
            kernels = ['rbf', 'linear', 'poly']

        svm_results = {}

        for kernel in kernels:
            logger.info("Training SVM with %s kernel...", kernel)

            svm_model = KernelSVMRegressor(kernel=kernel, auto_tune=auto_tune)

            # Fit model
            train_results = svm_model.fit(
                y=self.y_train.values,
                X_continuous=self.X_train.values,
                X_categorical=None,
                cv_folds=5
            )

            # Predictions
            y_train_pred = svm_model.predict(
                X_continuous=self.X_train.values,
                X_categorical=None,
                return_weighted=False
            )
            y_test_pred = svm_model.predict(
                X_continuous=self.X_test.values,
                X_categorical=None,
                return_weighted=False
            )

            # Metrics
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            test_mae = mean_absolute_error(self.y_test, y_test_pred)

            results = {
                'model': svm_model,
                'train_results': train_results,
                'train_r2': train_results['train_r2'],
                'test_r2': test_r2,
                'train_rmse': train_results['train_rmse'],
                'test_rmse': test_rmse,
                'train_mae': train_results['train_mae'],
                'test_mae': test_mae,
                'y_test_pred': y_test_pred,
                'y_train_pred': y_train_pred
            }

            svm_results[f'svm_{kernel}'] = results
            self.models[f'svm_{kernel}'] = svm_model
            self.model_results[f'svm_{kernel}'] = results

            logger.info("SVM (%s) training completed:", kernel)
            logger.info("  - Train R²: %.4f", results['train_r2'])
            logger.info("  - Test R²: %.4f", results['test_r2'])
            logger.info("  - Test RMSE: %.4f", results['test_rmse'])

        return svm_results

    def analyze_shap(self,
                    model_names: Optional[List[str]] = None,
                    max_samples: int = 500) -> Dict[str, Dict]:
        """
        Perform SHAP analysis on trained models

        Args:
            model_names: List of model names to analyze (None for all tree-based models)
            max_samples: Maximum samples for SHAP calculation

        Returns:
            Dictionary of SHAP results for each model
        """
        logger.info("=" * 60)
        logger.info("Performing SHAP analysis")
        logger.info("=" * 60)

        if not self.models:
            raise ValueError("No models trained. Train models first.")

        # Default to tree-based models
        if model_names is None:
            model_names = [name for name in self.models.keys() 
                          if name in ['xgboost', 'lightgbm']]

        for model_name in model_names:
            if model_name not in self.models:
                logger.warning("Model %s not found, skipping...", model_name)
                continue

            logger.info("Analyzing model: %s", model_name)

            model = self.models[model_name]

            # Create SHAP analyzer
            shap_analyzer = SHAPAnalyzer(model)

            try:
                # Create explainer
                shap_analyzer.create_explainer(self.X_test, explainer_type='auto')

                # Calculate SHAP values
                shap_values = shap_analyzer.calculate_shap_values(max_samples=max_samples)

                # Get feature importance
                feature_importance = shap_analyzer.get_feature_importance(importance_type='mean_abs')

                # Save results
                self.shap_results[model_name] = {
                    'analyzer': shap_analyzer,
                    'shap_values': shap_values,
                    'feature_importance': feature_importance
                }

                # Log top 10 features
                logger.info("Top 10 important features for %s:", model_name)
                for i, (feat, imp) in enumerate(list(feature_importance.items())[:10]):
                    logger.info("  %2d. %s: %.4f", i+1, feat, imp)

                # Save SHAP values
                shap_path = os.path.join(self.output_dir, f'shap_values_{model_name}.csv')
                shap_analyzer.export_shap_values(shap_path)

                # Create visualizations
                try:
                    # Summary plot (dot)
                    plot_path = os.path.join(self.output_dir, f'shap_summary_{model_name}.png')
                    shap_analyzer.summary_plot(plot_type='dot', save_path=plot_path)

                    # Feature importance plot
                    plot_path = os.path.join(self.output_dir, f'shap_importance_{model_name}.png')
                    shap_analyzer.plot_feature_importance(save_path=plot_path, top_n=20)
                except (IOError, RuntimeError, ValueError) as e:
                    logger.warning("Failed to create SHAP plots: %s", e)

            except (ValueError, RuntimeError, AttributeError) as e:
                logger.error("SHAP analysis failed for %s: %s", model_name, e)
                continue

        logger.info("SHAP analysis completed for %d models", len(self.shap_results))

        return self.shap_results

    def extract_nitrogen_impact(self) -> Dict[str, float]:
        """
        Extract nitrogen deposition impact from SHAP results

        Identifies features affected by nitrogen deposition and quantifies their contribution

        Returns:
            Dictionary with nitrogen-related feature contributions
        """
        logger.info("=" * 60)
        logger.info("Extracting nitrogen deposition impact")
        logger.info("=" * 60)

        if not self.shap_results:
            raise ValueError("No SHAP results available. Run analyze_shap() first.")

        # Features that are affected by nitrogen deposition
        nitrogen_affected_features = [
            'n_deposition',  # Direct nitrogen input
            'duration',  # Treatment duration
            # Soil properties affected by nitrogen
            'total_nitrogen_content',
            'cn_ratio',
            'organic_carbon_content',
            'ph_in_water',
            'cec_soil',
            'cec_eff',
            # Also consider PCA components that might capture N effects
        ]

        nitrogen_impacts = {}

        for model_name, results in self.shap_results.items():
            logger.info("Processing model: %s", model_name)

            feature_importance = results['feature_importance']

            # Extract nitrogen-related features
            model_impacts = {}
            total_importance = sum(feature_importance.values())

            for feature in nitrogen_affected_features:
                if feature in feature_importance:
                    importance = feature_importance[feature]
                    relative_importance = (importance / total_importance) * 100 if total_importance > 0 else 0
                    model_impacts[feature] = {
                        'absolute': importance,
                        'relative_pct': relative_importance
                    }
                    logger.info("  - %s: %.4f (%.2f%%)", feature, importance, relative_importance)

            # Also check for PCA components (they might capture N effects)
            for feature in feature_importance.keys():
                if feature.startswith('PC'):
                    model_impacts[feature] = {
                        'absolute': feature_importance[feature],
                        'relative_pct': (feature_importance[feature] / total_importance) * 100
                    }

            # Calculate total nitrogen contribution
            total_n_contribution = sum(imp['absolute'] for imp in model_impacts.values())
            total_n_contribution_pct = (total_n_contribution / total_importance) * 100 if total_importance > 0 else 0

            model_impacts['total_nitrogen_contribution'] = {
                'absolute': total_n_contribution,
                'relative_pct': total_n_contribution_pct
            }

            logger.info("Total nitrogen contribution: %.4f (%.2f%%)",
                       total_n_contribution, total_n_contribution_pct)

            nitrogen_impacts[model_name] = model_impacts

        # Save results
        impact_path = os.path.join(self.output_dir, 'nitrogen_impact_summary.json')
        import json
        with open(impact_path, 'w', encoding='utf-8') as f:
            json.dump(nitrogen_impacts, f, indent=2)
        logger.info("Saved nitrogen impact summary to %s", impact_path)

        return nitrogen_impacts

    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report of all analyses

        Returns:
            Summary report as string
        """
        logger.info("=" * 60)
        logger.info("Generating summary report")
        logger.info("=" * 60)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BIOMASS-NITROGEN RELATIONSHIP ANALYSIS SUMMARY")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Data summary
        report_lines.append("1. DATA SUMMARY")
        report_lines.append("-" * 80)
        if self.processed_data is not None:
            report_lines.append(f"Total samples: {len(self.processed_data)}")
            report_lines.append(f"  - Control samples: {len(self.processed_data[self.processed_data['group'] == 'control'])}")
            report_lines.append(f"  - Treatment samples: {len(self.processed_data[self.processed_data['group'] == 'treatment'])}")
        if self.X_train is not None:
            report_lines.append(f"Training samples: {len(self.X_train)}")
            report_lines.append(f"Testing samples: {len(self.X_test)}")
            report_lines.append(f"Number of features: {self.X_train.shape[1]}")
        report_lines.append("")

        # PCA summary
        if self.pca_analyzer is not None:
            report_lines.append("2. PCA DIMENSIONALITY REDUCTION")
            report_lines.append("-" * 80)
            _, cumulative_variance = self.pca_analyzer.get_explained_variance()
            report_lines.append(f"Number of components: {self.pca_analyzer.n_components}")
            report_lines.append(f"Total variance explained: {cumulative_variance[-1]*100:.2f}%")
            report_lines.append("")

        # Model performance
        report_lines.append("3. MODEL PERFORMANCE COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'Test RMSE':<12} {'Test MAE':<12}")
        report_lines.append("-" * 80)

        for model_name, results in self.model_results.items():
            report_lines.append(
                f"{model_name:<20} "
                f"{results['train_r2']:<12.4f} "
                f"{results['test_r2']:<12.4f} "
                f"{results['test_rmse']:<12.4f} "
                f"{results['test_mae']:<12.4f}"
            )
        report_lines.append("")

        # SHAP results
        if self.shap_results:
            report_lines.append("4. FEATURE IMPORTANCE (SHAP)")
            report_lines.append("-" * 80)

            for model_name, results in self.shap_results.items():
                report_lines.append(f"\nModel: {model_name}")
                report_lines.append(f"{'Rank':<6} {'Feature':<30} {'Importance':<12}")
                report_lines.append("-" * 50)

                feature_importance = results['feature_importance']
                for i, (feat, imp) in enumerate(list(feature_importance.items())[:10]):
                    report_lines.append(f"{i+1:<6} {feat:<30} {imp:<12.4f}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save report
        report_path = os.path.join(self.output_dir, 'analysis_summary.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info("Saved summary report to %s", report_path)

        # Also log to console
        logger.info("\n%s", report)

        return report

    def run_complete_analysis(self,
                             apply_nitrogen_impact: bool = True,
                             apply_pca: bool = True,
                             train_xgb: bool = True,
                             train_lgb: bool = True,
                             train_svm_models: bool = True,
                             perform_shap: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis pipeline

        Args:
            apply_nitrogen_impact: Whether to apply nitrogen impact on soil
            apply_pca: Whether to apply PCA
            train_xgb: Whether to train XGBoost
            train_lgb: Whether to train LightGBM
            train_svm_models: Whether to train SVM models
            perform_shap: Whether to perform SHAP analysis

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE BIOMASS-NITROGEN RELATIONSHIP ANALYSIS")
        logger.info("=" * 80)

        # Step 1: Load and preprocess data
        self.load_data()
        self.preprocess_data(apply_nitrogen_impact=apply_nitrogen_impact)

        # Step 2: Prepare features
        X, y = self.prepare_features(apply_pca=apply_pca, save_features=True)

        # Step 3: Split data
        self.split_data(X, y)

        # Step 4: Train models
        if train_xgb:
            self.train_xgboost()

        if train_lgb:
            self.train_lightgbm()

        if train_svm_models:
            self.train_svm(kernels=['rbf', 'linear'], auto_tune=False)

        # Step 5: SHAP analysis
        if perform_shap and (train_xgb or train_lgb):
            self.analyze_shap()
            self.extract_nitrogen_impact()

        # Step 6: Generate summary
        self.generate_summary_report()

        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETED SUCCESSFULLY")
        logger.info("Results saved to: %s", self.output_dir)
        logger.info("=" * 80)

        return {
            'processed_data': self.processed_data,
            'models': self.models,
            'model_results': self.model_results,
            'shap_results': self.shap_results
        }


def main():
    """
    Main entry point for biomass relationship analysis
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_data_path = os.path.join(project_root, 'data', 'sample.csv')
    soil_impact_data_path = os.path.join(project_root, 'data', 'soil_change_nitrogen.csv')
    output_dir = os.path.join(project_root, 'results', 'biomass_relationship')

    logger.info("Project root: %s", project_root)
    logger.info("Sample data: %s", sample_data_path)
    logger.info("Soil impact data: %s", soil_impact_data_path)
    logger.info("Output directory: %s", output_dir)

    # Create processor
    processor = BiomassRelationshipProcessor(
        sample_data_path=sample_data_path,
        soil_impact_data_path=soil_impact_data_path,
        output_dir=output_dir,
        n_components=None,
        variance_threshold=0.95,
        max_components=20,
        test_size=0.2,
        random_state=42
    )

    # Run complete analysis
    results = processor.run_complete_analysis(
        apply_nitrogen_impact=True,
        apply_pca=True,
        train_xgb=True,
        train_lgb=True,
        train_svm_models=True,
        perform_shap=True
    )

    logger.info("Analysis completed successfully!")

    return results


if __name__ == "__main__":
    main()
