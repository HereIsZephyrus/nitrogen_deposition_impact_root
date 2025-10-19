"""
Soil impact calculator - predicts soil property changes based on nitrogen addition
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Union, Literal
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class SoilCalculator:
    """
    Calculator for predicting soil properties based on nitrogen addition
    Uses data from soil_change.csv with various regression methods
    """

    def __init__(self, data_path: str, method: Literal['linear', 'spline', 'polynomial']):
        """
        Initialize soil calculator

        Args:
            data_path: Path to soil_change_wide.csv file
            method: Regression method - 'linear', 'spline', or 'polynomial'
        """
        self.data_path = data_path
        self.method = method
        self.data = None
        self.indicators = None
        self.models = {}

        self._load_data()
        self._build_models()

    def _load_data(self) -> None:
        """Load and parse soil change data from CSV"""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info("Loaded soil data: %d indicators", len(self.data))

            # Extract indicator names
            self.indicators = self.data['indicator'].tolist()

            # Log available indicators
            for ind in self.indicators:
                logger.debug("Available indicator: %s", ind)

        except Exception as e:
            logger.error("Error loading soil data: %s", e)
            raise

    def _extract_nitrogen_data(self, indicator: str) -> tuple:
        """
        Extract nitrogen addition levels and corresponding values for an indicator

        Args:
            indicator: Name of the soil indicator

        Returns:
            Tuple of (nitrogen_levels, mean_values, se_values)
        """
        # Get the row for this indicator
        row = self.data[self.data['indicator'] == indicator].iloc[0]

        # Extract nitrogen addition conditions (N0, N30, N60, N90, N120, N150)
        n_conditions = ['control_N0', 'N30', 'N60', 'N90', 'N120', 'N150']
        n_levels = [0, 30, 60, 90, 120, 150]  # kg N ha⁻¹ y⁻¹

        means = []
        ses = []

        for condition in n_conditions:
            mean_col = f'{condition}_mean'
            se_col = f'{condition}_se'

            if mean_col in row.index:
                means.append(row[mean_col])
                ses.append(row[se_col])
            else:
                logger.warning("Column %s not found for indicator %s", mean_col, indicator)

        return np.array(n_levels), np.array(means), np.array(ses)

    def _build_models(self) -> None:
        """Build regression models for all indicators"""
        logger.info("Building %s regression models...", self.method)

        for indicator in self.indicators:
            try:
                n_levels, means, ses = self._extract_nitrogen_data(indicator)

                if self.method == 'linear':
                    # Linear regression
                    model = LinearRegression()
                    model.fit(n_levels.reshape(-1, 1), means)
                    self.models[indicator] = {
                        'type': 'linear',
                        'model': model,
                        'n_levels': n_levels,
                        'means': means,
                        'ses': ses
                    }

                elif self.method == 'spline':
                    # Cubic spline interpolation
                    spline = UnivariateSpline(n_levels, means, s=0, k=min(3, len(n_levels)-1))
                    self.models[indicator] = {
                        'type': 'spline',
                        'model': spline,
                        'n_levels': n_levels,
                        'means': means,
                        'ses': ses
                    }

                elif self.method == 'polynomial':
                    # Polynomial regression (degree 2)
                    coeffs = np.polyfit(n_levels, means, deg=2)
                    poly = np.poly1d(coeffs)
                    self.models[indicator] = {
                        'type': 'polynomial',
                        'model': poly,
                        'n_levels': n_levels,
                        'means': means,
                        'ses': ses,
                        'coeffs': coeffs
                    }

                logger.debug("Built model for %s", indicator)

            except Exception as e:
                logger.error("Error building model for %s: %s", indicator, e)
                continue

        logger.info("Built %d models successfully", len(self.models))

    def predict(self, indicator: str, n_addition: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Predict soil property value for given nitrogen addition level(s)

        Args:
            indicator: Name of the soil indicator
            n_addition: Nitrogen addition level(s) in kg N ha⁻¹ y⁻¹

        Returns:
            Predicted value(s) for the indicator
        """
        if indicator not in self.models:
            raise ValueError(f"No model available for indicator: {indicator}")

        model_data = self.models[indicator]
        model = model_data['model']

        # Ensure input is array
        n_addition = np.atleast_1d(n_addition)

        # Clip to valid range
        n_min, n_max = model_data['n_levels'].min(), model_data['n_levels'].max()
        n_clipped = np.clip(n_addition, n_min, n_max)

        if np.any(n_addition != n_clipped):
            logger.warning("Nitrogen levels clipped to range [%d, %d]", n_min, n_max)

        # Predict based on model type
        if model_data['type'] == 'linear':
            predictions = model.predict(n_clipped.reshape(-1, 1))
        elif model_data['type'] == 'spline':
            predictions = model(n_clipped)
        elif model_data['type'] == 'polynomial':
            predictions = model(n_clipped)

        # Return scalar if input was scalar
        if len(predictions) == 1:
            return float(predictions[0])
        return predictions

    def predict_all(self, n_addition: float) -> Dict[str, float]:
        """
        Predict all soil indicators for a given nitrogen addition level

        Args:
            n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹

        Returns:
            Dictionary mapping indicator names to predicted values
        """
        predictions = {}

        for indicator in self.indicators:
            try:
                predictions[indicator] = self.predict(indicator, n_addition)
            except Exception as e:
                logger.error("Error predicting %s: %s", indicator, e)
                predictions[indicator] = None

        return predictions

    def predict_soil_variance(self, soil_variance, n_addition: float, duration: float = None):
        """
        Predict soil changes based on ImpactedSoilVariance input

        Args:
            soil_variance: ImpactedSoilVariance object or dictionary of soil properties
            n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹
            duration: Duration in years (optional, used for time-based calculations)

        Returns:
            ImpactedSoilVariance object with predicted changes
        """
        from .variance import ImpactedSoilVariance

        # Get predictions for all indicators
        predictions = self.predict_all(n_addition)

        # Map predictions to ImpactedSoilVariance structure
        return ImpactedSoilVariance(
            soil_ph=predictions.get('soil_ph', 0.0),
            total_nitrogen=predictions.get('total_nitrogen', 0.0)
        )

    def get_unit(self, indicator: str) -> str:
        """Get the unit for a specific indicator"""
        row = self.data[self.data['indicator'] == indicator]
        if len(row) > 0:
            return row['unit'].iloc[0]
        return 'unknown'

    def get_control_value(self, indicator: str) -> float:
        """Get the control (N0) value for an indicator"""
        n_levels, means, _ = self._extract_nitrogen_data(indicator)
        return means[0]  # First value is control (N0)

    def calculate_change(self, indicator: str, n_addition: float, relative: bool = False) -> float:
        """
        Calculate change from control for an indicator

        Args:
            indicator: Name of the soil indicator
            n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹
            relative: If True, return relative change (%), otherwise absolute change

        Returns:
            Change from control value
        """
        control = self.get_control_value(indicator)
        predicted = self.predict(indicator, n_addition)

        if relative:
            return ((predicted - control) / control) * 100
        else:
            return predicted - control

    def __repr__(self) -> str:
        """String representation"""
        return (f"SoilCalculator(method='{self.method}', "
                f"indicators={len(self.indicators)}, "
                f"models={len(self.models)})")

def transfer_N_input(n_addition: float, n_type: str) -> float:
    """
    Transfer nitrogen input to kg N ha⁻¹ y⁻¹

    Args:
        n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹
        n_type: Nitrogen type (N, NO3, NH4)

    Returns:
        Nitrogen addition level in kg N ha⁻¹ y⁻¹
    """
    if n_type == 'N':
        return n_addition
    elif n_type == 'NO3':
        # NaNO3: 85 g/mol, N: 14 g/mol
        # 85/14 = 6.07
        return n_addition / 6.07
    elif n_type == 'NH4':
        # (NH4)2SO4: 132 g/mol, 2N: 28 g/mol
        # 132/28 = 4.71
        return n_addition / 4.71
    else:
        raise ValueError(f"Invalid nitrogen type: {n_type}")
