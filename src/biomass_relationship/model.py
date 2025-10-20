"""
Model wrapper for biomass prediction

Encapsulates trained models with all necessary metadata and components
for making predictions on new data.
"""
import os
import logging
from typing import Any, Dict, Optional
import pickle
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class BiomassModel:
    """
    Complete biomass prediction model wrapper
    
    Encapsulates:
    - Trained prediction model (NLS/SVM/Decision Tree)
    - PCA transformer for feature dimensionality reduction
    - Nitrogen impact calculator for soil changes
    - Performance metrics and metadata
    - All parameters needed for prediction
    """
    
    def __init__(
        self,
        model_name: str,
        model_type: str,
        trained_model: Any,
        climate_group: int,
        performance_metrics: Dict[str, float],
        pca_analyzer: Optional[Any] = None,
        n_impact_calculator: Optional[Any] = None,
        feature_names: Optional[list] = None,
        training_config: Optional[Dict[str, Any]] = None,
        model_variant: Optional[str] = None
    ):
        """
        Initialize biomass prediction model
        
        Args:
            model_name: Full model name (e.g., "NLS_AdditiveModel")
            model_type: Type of model ("NLS", "SVM", or "DecisionTree")
            trained_model: The actual trained model object
            climate_group: Climate group ID this model belongs to
            performance_metrics: Dict with 'r2', 'rmse', 'mae', 'mape', etc.
            pca_analyzer: PCAnalyzer instance for feature transformation
            n_impact_calculator: NitrogenToSoilInfluencer for soil impact
            feature_names: List of feature names used in training
            training_config: Dict with training parameters
            model_variant: Specific variant name (e.g., "rbf" for SVM kernel)
        """
        self.model_name = model_name
        self.model_type = model_type
        self.trained_model = trained_model
        self.climate_group = climate_group
        self.performance_metrics = performance_metrics
        self.pca_analyzer = pca_analyzer
        self.n_impact_calculator = n_impact_calculator
        self.feature_names = feature_names or []
        self.training_config = training_config or {}
        self.model_variant = model_variant
        
        # Metadata
        self.created_at = datetime.now().isoformat()
        self.version = "1.0"
        
        logger.info(f"Initialized BiomassModel: {model_name} for group {climate_group}")
        logger.info(f"  Performance: R² = {performance_metrics.get('r2', 'N/A'):.4f}")
    
    def predict(
        self,
        climate_data: np.ndarray,
        categorical_data: np.ndarray,
        average_biomass: np.ndarray,
        nitrogen_addition: np.ndarray,
        nitrogen_duration: Optional[np.ndarray] = None,
        is_control: bool = False
    ) -> np.ndarray:
        """
        Make biomass predictions on new data
        
        Args:
            climate_data: Raw climate/environmental features (n_samples, n_features)
            categorical_data: Categorical features (one-hot encoded)
            average_biomass: Average biomass values (standardized)
            nitrogen_addition: N addition rates (kg N/ha/yr)
            nitrogen_duration: Duration of N addition (years), required if not control
            is_control: If True, predict control group (no N addition)
        
        Returns:
            Predicted log biomass values
        """
        if not is_control and nitrogen_duration is None:
            raise ValueError("nitrogen_duration required when is_control=False")
        
        # Apply nitrogen impact to soil if not control
        if is_control or np.all(nitrogen_addition == 0):
            processed_climate = climate_data
            n_addtion_total = np.zeros(len(climate_data))
        else:
            if self.n_impact_calculator is None:
                logger.warning("No nitrogen impact calculator provided, using raw climate data")
                processed_climate = climate_data
            else:
                processed_climate = self.n_impact_calculator.impact(
                    climate_data,
                    nitrogen_addition,
                    nitrogen_duration
                )
            n_addtion_total = nitrogen_addition * nitrogen_duration
        
        # Apply PCA transformation
        if self.pca_analyzer is None:
            logger.warning("No PCA analyzer provided, using raw climate data")
            transformed_data = processed_climate
        else:
            transformed_data = self.pca_analyzer.transform(processed_climate)
        
        # Combine all features
        input_features = np.column_stack([
            transformed_data,
            categorical_data,
            average_biomass.reshape(-1, 1)
        ])
        
        # Add nitrogen addition as first feature for prediction
        X_full = np.column_stack([n_addtion_total, input_features])
        
        # Make prediction
        predictions = self.trained_model.predict(X_full)
        
        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions
    
    def predict_both(
        self,
        climate_data: np.ndarray,
        categorical_data: np.ndarray,
        average_biomass: np.ndarray,
        nitrogen_addition: np.ndarray,
        nitrogen_duration: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Predict both control and nitrogen-addition biomass
        
        Args:
            climate_data: Raw climate/environmental features
            categorical_data: Categorical features
            average_biomass: Average biomass values
            nitrogen_addition: N addition rates
            nitrogen_duration: Duration of N addition
        
        Returns:
            Dict with 'control' and 'treatment' predictions
        """
        control_pred = self.predict(
            climate_data=climate_data,
            categorical_data=categorical_data,
            average_biomass=average_biomass,
            nitrogen_addition=nitrogen_addition,
            nitrogen_duration=nitrogen_duration,
            is_control=True
        )
        
        treatment_pred = self.predict(
            climate_data=climate_data,
            categorical_data=categorical_data,
            average_biomass=average_biomass,
            nitrogen_addition=nitrogen_addition,
            nitrogen_duration=nitrogen_duration,
            is_control=False
        )
        
        return {
            'control': control_pred,
            'treatment': treatment_pred,
            'difference': treatment_pred - control_pred,
            'percent_change': (treatment_pred - control_pred) / control_pred * 100
        }
    
    def save(self, path: str):
        """
        Save complete model to file
        
        Args:
            path: Path to save the model (.pkl file)
        """
        # Package everything into a dict
        model_package = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'trained_model': self.trained_model,
            'climate_group': self.climate_group,
            'performance_metrics': self.performance_metrics,
            'pca_analyzer': self.pca_analyzer,
            'n_impact_calculator': self.n_impact_calculator,
            'feature_names': self.feature_names,
            'training_config': self.training_config,
            'model_variant': self.model_variant,
            'created_at': self.created_at,
            'version': self.version
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"Model saved to: {path}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Group: {self.climate_group}")
        logger.info(f"  R²: {self.performance_metrics.get('r2', 'N/A'):.4f}")
    
    @classmethod
    def load(cls, path: str) -> 'BiomassModel':
        """
        Load model from file
        
        Args:
            path: Path to the saved model file
        
        Returns:
            BiomassModel instance
        """
        with open(path, 'rb') as f:
            model_package = pickle.load(f)
        
        # Create instance from saved data
        instance = cls(
            model_name=model_package['model_name'],
            model_type=model_package['model_type'],
            trained_model=model_package['trained_model'],
            climate_group=model_package['climate_group'],
            performance_metrics=model_package['performance_metrics'],
            pca_analyzer=model_package.get('pca_analyzer'),
            n_impact_calculator=model_package.get('n_impact_calculator'),
            feature_names=model_package.get('feature_names', []),
            training_config=model_package.get('training_config', {}),
            model_variant=model_package.get('model_variant')
        )
        
        instance.created_at = model_package.get('created_at', 'unknown')
        instance.version = model_package.get('version', '1.0')
        
        logger.info(f"Model loaded from: {path}")
        logger.info(f"  Model: {instance.model_name}")
        logger.info(f"  Group: {instance.climate_group}")
        logger.info(f"  R²: {instance.performance_metrics.get('r2', 'N/A'):.4f}")
        
        return instance
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get complete model information as a dictionary
        
        Returns:
            Dict with all model metadata
        """
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_variant': self.model_variant,
            'climate_group': self.climate_group,
            'performance_metrics': self.performance_metrics,
            'feature_names': self.feature_names,
            'training_config': self.training_config,
            'created_at': self.created_at,
            'version': self.version,
            'has_pca': self.pca_analyzer is not None,
            'has_n_calculator': self.n_impact_calculator is not None
        }
    
    def summary(self) -> str:
        """
        Get human-readable summary of the model
        
        Returns:
            Formatted string with model information
        """
        lines = [
            "=" * 70,
            f"Biomass Prediction Model Summary",
            "=" * 70,
            f"Model Name: {self.model_name}",
            f"Model Type: {self.model_type}",
            f"Climate Group: {self.climate_group}",
            "",
            "Performance Metrics (LOOCV):",
        ]
        
        for metric, value in self.performance_metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {metric.upper()}: {value:.4f}")
            else:
                lines.append(f"  {metric.upper()}: {value}")
        
        lines.extend([
            "",
            "Components:",
            f"  PCA Analyzer: {'Yes' if self.pca_analyzer else 'No'}",
            f"  N Impact Calculator: {'Yes' if self.n_impact_calculator else 'No'}",
            f"  Feature Count: {len(self.feature_names)}",
            "",
            f"Created: {self.created_at}",
            f"Version: {self.version}",
            "=" * 70
        ])
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"BiomassModel(name='{self.model_name}', group={self.climate_group}, "
                f"R²={self.performance_metrics.get('r2', 'N/A'):.4f})")
