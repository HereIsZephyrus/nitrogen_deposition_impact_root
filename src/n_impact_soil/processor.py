from typing import Literal, List
import numpy as np
import pandas as pd
from .nitrogen_calculator import NitrogenCalculator
from .year_calculator import YearCalculator
from .variance import ImpactedSoilVariance

class NitrogenToSoilInfluencer:
    def __init__(self,
        nitrogen_impact_data_path: str,
        time_impact_data_path: str,
        nitrogen_method: Literal['linear', 'spline', 'polynomial'] = 'spline',
        time_method: Literal['linear', 'spline', 'polynomial'] = 'linear'):
        self.nitrogen_impact_data_path = nitrogen_impact_data_path
        self.time_impact_data_path = time_impact_data_path
        self.nitrogen_calculator = NitrogenCalculator(data_path=nitrogen_impact_data_path, method=nitrogen_method)
        self.year_calculator = YearCalculator(data_path=time_impact_data_path, method=time_method)

    def impact(self, raw_data: pd.DataFrame, n_addition: np.ndarray, duration: np.ndarray) -> pd.DataFrame:
        """
        Predict soil change for a given nitrogen addition and duration

        Args:
            raw_data: Raw data DataFrame
            n_addition: Nitrogen addition level in kg N ha⁻¹ y⁻¹
            duration: Duration in years
        Returns:
            Modified raw data DataFrame with soil changes applied
        """
        # Extract initial soil values
        initial_variance_list = self._data_input_adapter(raw_data)
        final_variance_list = []
        for variance, addition, dur in zip(initial_variance_list, n_addition, duration):
            # Calculate total nitrogen input
            total_n_addition = addition * dur

            # Apply nitrogen impact first
            after_nitrogen_variance = ImpactedSoilVariance(
                soil_ph=self.nitrogen_calculator.predict('soil_ph', variance.soil_ph, total_n_addition),
                total_nitrogen=self.nitrogen_calculator.predict('total_nitrogen', variance.total_nitrogen, total_n_addition)
            )

            # Then apply time impact on the nitrogen-affected values
            #final_variance = ImpactedSoilVariance(
            #    soil_ph=self.year_calculator.predict('soil_ph', after_nitrogen.soil_ph, duration),
            #    total_nitrogen=self.year_calculator.predict('total_nitrogen', after_nitrogen.total_nitrogen, duration)
            #)
            final_variance_list.append(after_nitrogen_variance)

        return self._data_output_adapter(raw_data, final_variance_list)

    def _data_input_adapter(self, raw_data: pd.DataFrame) -> List[ImpactedSoilVariance]:
        """
        Convert raw data array to ImpactedSoilVariance object

        Args:
            raw_data: Raw data
        Returns:
            ImpactedSoilVariance object containing relevant soil properties
        """
        ph_in_water = raw_data.ph_in_water
        total_nitrogen_content = raw_data.total_nitrogen_content

        variance_list = []

        for row in raw_data.itertuples():
            variance_list.append(ImpactedSoilVariance(
                soil_ph=float(row.ph_in_water),
                total_nitrogen=float(row.total_nitrogen_content)
            ))

        return variance_list

    def _data_output_adapter(self, raw_data: pd.DataFrame, final_variance_list: List[ImpactedSoilVariance]) -> pd.DataFrame:
        """
        Apply soil changes back to the raw data DataFrame

        Args:
            raw_data: Original raw data DataFrame
            final_variance_list: List of ImpactedSoilVariance with final values
        Returns:
            Modified raw data DataFrame with soil changes applied
        """
        modified_data = raw_data.copy()
        for idx, final_variance in enumerate(final_variance_list):
            modified_data.iloc[idx, modified_data.columns.get_loc('ph_in_water')] = final_variance.soil_ph
            modified_data.iloc[idx, modified_data.columns.get_loc('total_nitrogen_content')] = final_variance.total_nitrogen

        return modified_data
