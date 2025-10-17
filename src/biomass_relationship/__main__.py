"""
Biomass Relationship Statistical Analysis Framework
Modular statistical framework based on XGBoost notebook
"""
import logging
import datetime
from .processor import main

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)


if __name__ == "__main__":
    main()
