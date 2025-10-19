"""
Nitrogen impact calculator - predicts soil property change rates based on nitrogen addition
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Literal
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class NitrogenCalculator:
    """
    Calculator for predicting soil property change rates based on nitrogen addition
    Uses data from soil_change_nitrogen.csv with control_N0 as baseline
    """

    def __init__(self, data_path: str, method: Literal['linear', 'spline', 'polynomial'] = 'spline'):
        """
        Initialize nitrogen calculator

        Args:
            data_path: Path to soil_change_nitrogen.csv file
            method: Regression method - 'linear', 'spline', or 'polynomial'
        """
        self.data_path = data_path
        self.method = method
        self.data = None
        self.indicators = None
        self.models = {}
        self.baseline_values = {}  # Store control_N0 values

        self._load_data()
        self._build_models()

    def _load_data(self) -> None:
        """Load and parse soil change data from CSV"""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info("Loaded nitrogen impact data: %d indicators", len(self.data))

            # Extract indicator names
            self.indicators = self.data['indicator'].tolist()

            # Log available indicators
            for ind in self.indicators:
                logger.debug("Available indicator: %s", ind)

        except Exception as e:
            logger.error("Error loading nitrogen data: %s", e)
            raise

    def _extract_change_rates(self, indicator: str) -> tuple:
        """
        Extract nitrogen addition levels and corresponding change rates for an indicator

        Args:
            indicator: Name of the soil indicator

        Returns:
            Tuple of (nitrogen_levels, change_rates)
        """
        # Get the row for this indicator
        row = self.data[self.data['indicator'] == indicator].iloc[0]

        # Extract nitrogen addition conditions (N0, N30, N60, N90, N120, N150)
        n_conditions = ['control_N0', 'N30', 'N60', 'N90', 'N120', 'N150']
        n_levels = [0, 30, 60, 90, 120, 150]  # kg N ha⁻¹ y⁻¹

        # Get baseline (control_N0) value
        baseline = row['control_N0_mean']
        self.baseline_values[indicator] = baseline

        change_rates = []
        for condition in n_conditions:
            mean_col = f'{condition}_mean'

            if mean_col in row.index:
                value = row[mean_col]
                # Calculate change rate relative to control
                change_rate = (value - baseline) / baseline if baseline != 0 else 0
                change_rates.append(change_rate)
            else:
                logger.warning("Column %s not found for indicator %s", mean_col, indicator)
                change_rates.append(0)

        return np.array(n_levels), np.array(change_rates)

    def _build_models(self) -> None:
        """Build regression models for all indicators"""
        logger.info("Building %s regression models for nitrogen impact...", self.method)

        for indicator in self.indicators:
            try:
                n_levels, change_rates = self._extract_change_rates(indicator)

                if self.method == 'linear':
                    # Linear regression
                    model = LinearRegression()
                    model.fit(n_levels.reshape(-1, 1), change_rates)
                    self.models[indicator] = {
                        'type': 'linear',
                        'model': model,
                        'n_levels': n_levels,
                        'change_rates': change_rates
                    }

                elif self.method == 'spline':
                    # Cubic spline interpolation
                    spline = UnivariateSpline(n_levels, change_rates, s=0, k=min(3, len(n_levels)-1))
                    self.models[indicator] = {
                        'type': 'spline',
                        'model': spline,
                        'n_levels': n_levels,
                        'change_rates': change_rates
                    }

                elif self.method == 'polynomial':
                    # Polynomial regression (degree 2)
                    coeffs = np.polyfit(n_levels, change_rates, deg=2)
                    poly = np.poly1d(coeffs)
                    self.models[indicator] = {
                        'type': 'polynomial',
                        'model': poly,
                        'n_levels': n_levels,
                        'change_rates': change_rates,
                        'coeffs': coeffs
                    }

                logger.debug("Built nitrogen model for %s (baseline=%.2f)", 
                           indicator, self.baseline_values[indicator])

            except Exception as e:
                logger.error("Error building model for %s: %s", indicator, e)
                continue

        logger.info("Built %d nitrogen models successfully", len(self.models))

    def predict(self, indicator: str, initial_value: float, n_addition: float) -> float:
        """
        Predict soil property value after nitrogen addition

        Args:
            indicator: Name of the soil indicator
            initial_value: Initial value of the soil property
            n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹

        Returns:
            Predicted value after nitrogen addition (initial_value * (1 + change_rate))
        """
        if indicator not in self.models:
            raise ValueError(f"No model available for indicator: {indicator}")

        model_data = self.models[indicator]
        model = model_data['model']

        # Clip to valid range
        n_min, n_max = model_data['n_levels'].min(), model_data['n_levels'].max()
        n_clipped = np.clip(n_addition, n_min, n_max)

        if n_addition != n_clipped:
            logger.warning("Nitrogen level clipped from %.1f to %.1f", n_addition, n_clipped)

        # Predict change rate based on model type
        change_rate = 0.0
        if model_data['type'] == 'linear':
            change_rate = model.predict([[n_clipped]])[0]
        elif model_data['type'] == 'spline':
            change_rate = float(model(n_clipped))
        elif model_data['type'] == 'polynomial':
            change_rate = float(model(n_clipped))

        # Apply change rate to initial value
        final_value = initial_value * (1 + change_rate)

        logger.debug("Nitrogen impact for %s: initial=%.4f, N=%.1f, rate=%.4f, final=%.4f",
                    indicator, initial_value, n_addition, change_rate, final_value)

        return final_value

    def get_change_rate(self, indicator: str, n_addition: float) -> float:
        """
        Get the change rate for a given nitrogen addition level

        Args:
            indicator: Name of the soil indicator
            n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹

        Returns:
            Change rate (as decimal, e.g., 0.1 means 10% increase)
        """
        if indicator not in self.models:
            raise ValueError(f"No model available for indicator: {indicator}")

        model_data = self.models[indicator]
        model = model_data['model']

        # Clip to valid range
        n_clipped = np.clip(n_addition, 
                           model_data['n_levels'].min(), 
                           model_data['n_levels'].max())

        # Predict change rate
        change_rate = 0.0
        if model_data['type'] == 'linear':
            change_rate = model.predict([[n_clipped]])[0]
        elif model_data['type'] == 'spline':
            change_rate = float(model(n_clipped))
        elif model_data['type'] == 'polynomial':
            change_rate = float(model(n_clipped))

        return change_rate

    def get_baseline_value(self, indicator: str) -> float:
        """Get the baseline (control_N0) value for an indicator"""
        return self.baseline_values.get(indicator, 0.0)

    def predict_all(self, initial_values: Dict[str, float], n_addition: float) -> Dict[str, float]:
        """
        Predict all soil indicators after nitrogen addition

        Args:
            initial_values: Dictionary mapping indicator names to initial values
            n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹

        Returns:
            Dictionary mapping indicator names to predicted values
        """
        predictions = {}

        for indicator in self.indicators:
            try:
                initial_value = initial_values.get(indicator, self.baseline_values.get(indicator, 0))
                predictions[indicator] = self.predict(indicator, initial_value, n_addition)
            except Exception as e:
                logger.error("Error predicting %s: %s", indicator, e)
                predictions[indicator] = initial_values.get(indicator, 0)

        return predictions

    def __repr__(self) -> str:
        """String representation"""
        return (f"NitrogenCalculator(method='{self.method}', "
                f"indicators={len(self.indicators)}, "
                f"models={len(self.models)})")

