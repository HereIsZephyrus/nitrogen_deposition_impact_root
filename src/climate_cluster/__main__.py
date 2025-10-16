#!/usr/bin/env python3
"""
Climate Clustering Main Entry Point

This module provides the main entry point for running the complete climate clustering pipeline.
It processes climate data through standardization, sample clustering, global clustering,
and secondary clustering steps.

Usage:
    python -m climate_cluster
    python -m src.climate_cluster

Example with custom parameters:
    python -m src.climate_cluster --sample-k 5 --confidence 0.75 --output results/
"""
import logging
import datetime
from .processor import main

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)

if __name__ == "__main__":
    main()
