"""
Biomass Relationship Statistical Analysis Framework
Modular statistical framework based on XGBoost notebook
"""
import argparse
import logging
import datetime
import matplotlib
from .processor import train

matplotlib.use('Agg')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistic-folder", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--climate", type=str, required=True)
    args = parser.parse_args()
    train(
        statistic_folder_path=args.statistic_folder,
        climate_dir=args.climate,
        output_dir=args.output
    )
