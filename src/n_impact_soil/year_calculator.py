"""
Year impact calculator - predicts soil property change rates based on time duration
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Literal
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class YearCalculator:
    """
    Calculator for predicting soil property change rates based on time duration
    Uses data from soil_change_year.csv with year_1 as baseline
    """

    def __init__(self, data_path: str, method: Literal['linear', 'spline', 'polynomial'] = 'linear'):
        """
        Initialize year calculator

        Args:
            data_path: Path to soil_change_year.csv file
            method: Regression method - 'linear', 'spline', or 'polynomial'
        """
        self.data_path = data_path
        self.method = method
        self.data = None
        self.indicators = None
        self.models = {}
        self.baseline_values = {}  # Store year_1 values

        self._load_data()
        self._build_models()

    def _load_data(self) -> None:
        """Load and parse soil change data from CSV"""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info("Loaded year impact data: %d indicators", len(self.data))

            # Extract indicator names
            self.indicators = self.data['indicator'].tolist()

            # Log available indicators
            for ind in self.indicators:
                logger.debug("Available indicator: %s", ind)

        except Exception as e:
            logger.error("Error loading year data: %s", e)
            raise

    def _extract_change_rates(self, indicator: str) -> tuple:
        """
        Extract year durations and corresponding change rates for an indicator

        Args:
            indicator: Name of the soil indicator

        Returns:
            Tuple of (years, change_rates)
        """
        # Get the row for this indicator
        row = self.data[self.data['indicator'] == indicator].iloc[0]

        # Extract year conditions (year_1, year_2, year_3)
        year_conditions = ['year_1', 'year_2', 'year_3']
        years = [1, 2, 3]  # Duration in years

        # Get baseline (year_1) value
        baseline = row['year_1_mean']
        self.baseline_values[indicator] = baseline

        change_rates = []
        for condition in year_conditions:
            mean_col = f'{condition}_mean'

            if mean_col in row.index:
                value = row[mean_col]
                # Calculate change rate relative to year_1
                change_rate = (value - baseline) / baseline if baseline != 0 else 0
                change_rates.append(change_rate)
            else:
                logger.warning("Column %s not found for indicator %s", mean_col, indicator)
                change_rates.append(0)

        return np.array(years, dtype=float), np.array(change_rates)

    def _build_models(self) -> None:
        """Build regression models for all indicators"""
        logger.info("Building %s regression models for year impact...", self.method)

        for indicator in self.indicators:
            try:
                years, change_rates = self._extract_change_rates(indicator)

                if self.method == 'linear':
                    # Linear regression
                    model = LinearRegression()
                    model.fit(years.reshape(-1, 1), change_rates)
                    self.models[indicator] = {
                        'type': 'linear',
                        'model': model,
                        'years': years,
                        'change_rates': change_rates
                    }

                elif self.method == 'spline':
                    # Linear spline (k=1) for only 3 data points
                    spline = UnivariateSpline(years, change_rates, s=0, k=min(1, len(years)-1))
                    self.models[indicator] = {
                        'type': 'spline',
                        'model': spline,
                        'years': years,
                        'change_rates': change_rates
                    }

                elif self.method == 'polynomial':
                    # Polynomial regression (degree 2)
                    coeffs = np.polyfit(years, change_rates, deg=2)
                    poly = np.poly1d(coeffs)
                    self.models[indicator] = {
                        'type': 'polynomial',
                        'model': poly,
                        'years': years,
                        'change_rates': change_rates,
                        'coeffs': coeffs
                    }

                logger.debug("Built year model for %s (baseline=%.2f)", 
                           indicator, self.baseline_values[indicator])

            except Exception as e:
                logger.error("Error building model for %s: %s", indicator, e)
                continue

        logger.info("Built %d year models successfully", len(self.models))

    def predict(self, indicator: str, initial_value: float, duration: float) -> float:
        """
        Predict soil property value after time duration

        Args:
            indicator: Name of the soil indicator
            initial_value: Initial value of the soil property
            duration: Duration in years

        Returns:
            Predicted value after duration (initial_value * (1 + change_rate))
        """
        if indicator not in self.models:
            raise ValueError(f"No model available for indicator: {indicator}")

        model_data = self.models[indicator]
        model = model_data['model']

        # Clip to valid range (typically 1-3 years)
        year_min, year_max = model_data['years'].min(), model_data['years'].max()
        duration_clipped = np.clip(duration, year_min, year_max)

        if duration != duration_clipped:
            logger.warning("Duration clipped from %.1f to %.1f years", duration, duration_clipped)

        # Predict change rate based on model type
        change_rate = 0.0
        if model_data['type'] == 'linear':
            change_rate = model.predict([[duration_clipped]])[0]
        elif model_data['type'] == 'spline':
            change_rate = float(model(duration_clipped))
        elif model_data['type'] == 'polynomial':
            change_rate = float(model(duration_clipped))

        # Apply change rate to initial value
        final_value = initial_value * (1 + change_rate)

        logger.debug("Year impact for %s: initial=%.4f, years=%.1f, rate=%.4f, final=%.4f",
                    indicator, initial_value, duration, change_rate, final_value)

        return final_value

    def get_change_rate(self, indicator: str, duration: float) -> float:
        """
        Get the change rate for a given duration

        Args:
            indicator: Name of the soil indicator
            duration: Duration in years

        Returns:
            Change rate (as decimal, e.g., 0.1 means 10% increase)
        """
        if indicator not in self.models:
            raise ValueError(f"No model available for indicator: {indicator}")

        model_data = self.models[indicator]
        model = model_data['model']

        # Clip to valid range
        duration_clipped = np.clip(duration, 
                                   model_data['years'].min(), 
                                   model_data['years'].max())

        # Predict change rate
        change_rate = 0.0
        if model_data['type'] == 'linear':
            change_rate = model.predict([[duration_clipped]])[0]
        elif model_data['type'] == 'spline':
            change_rate = float(model(duration_clipped))
        elif model_data['type'] == 'polynomial':
            change_rate = float(model(duration_clipped))

        return change_rate

    def get_baseline_value(self, indicator: str) -> float:
        """Get the baseline (year_1) value for an indicator"""
        return self.baseline_values.get(indicator, 0.0)

    def predict_all(self, initial_values: Dict[str, float], duration: float) -> Dict[str, float]:
        """
        Predict all soil indicators after time duration

        Args:
            initial_values: Dictionary mapping indicator names to initial values
            duration: Duration in years

        Returns:
            Dictionary mapping indicator names to predicted values
        """
        predictions = {}

        for indicator in self.indicators:
            try:
                initial_value = initial_values.get(indicator, self.baseline_values.get(indicator, 0))
                predictions[indicator] = self.predict(indicator, initial_value, duration)
            except Exception as e:
                logger.error("Error predicting %s: %s", indicator, e)
                predictions[indicator] = initial_values.get(indicator, 0)

        return predictions

    def __repr__(self) -> str:
        """String representation"""
        return (f"YearCalculator(method='{self.method}', "
                f"indicators={len(self.indicators)}, "
                f"models={len(self.models)})")

