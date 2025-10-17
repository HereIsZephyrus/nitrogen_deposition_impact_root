"""
N Impact Soil Main Entry Point

This module provides the main entry point for running the complete N impact soil pipeline.
It processes soil data through standardization, sample clustering then global clustering,
and secondary clustering steps.

Example with custom parameters:
    python -m src.n_impact_soil --control-folder ./data/ --n-addition 100 --n-type N
"""
import os
import logging
import datetime
import argparse
from .processor import predict_soil_change
from .calculator import SoilCalculator

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)


def main(control_folder: str, input_N_addition: float, input_N_type: str):
    """
    Main function to predict soil change
    """
    year_calculator = SoilCalculator(data_path=os.path.join(control_folder, 'soil_change_year.csv'), method='linear')
    season_calculator = SoilCalculator(data_path=os.path.join(control_folder, 'soil_change_season.csv'), method='linear')
    nitrogen_calculator = SoilCalculator(data_path=os.path.join(control_folder, 'soil_change_month.csv'), method='linear')
    year_result = predict_soil_change(year_calculator, n_addition=input_N_addition, n_type=input_N_type)
    season_result = predict_soil_change(season_calculator, n_addition=input_N_addition, n_type=input_N_type)
    nitrogen_result = predict_soil_change(nitrogen_calculator, n_addition=input_N_addition, n_type=input_N_type)
    print(year_result)
    print(season_result)
    print(nitrogen_result)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--control-folder", type=str, required=True)
    args.add_argument("--n-addition", type=float, required=True)
    args.add_argument("--n-type", type=str, required=True)
    args = args.parse_args()
    main(args.control_folder, args.n_addition, args.n_type)
