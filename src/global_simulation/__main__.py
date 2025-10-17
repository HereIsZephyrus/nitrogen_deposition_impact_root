import logging
import datetime
import argparse
from .processor import main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--target-resolution", type=float, required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.target_resolution)
