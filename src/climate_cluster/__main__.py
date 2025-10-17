#!/usr/bin/env python3
"""
Climate Clustering Main Entry Point

This module provides the main entry point for running the complete climate clustering pipeline.
It processes climate data through standardization, sample clustering then global clustering,
and secondary clustering steps.

Example with custom parameters:
    python -m src.climate_cluster --sample-k 5 --confidence 0.80 --mask ./data/mask.shp --output ./results/ --climate ./climate/ --sample ./sample.csv
"""
import logging
import datetime
import argparse
from .processor import main

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'logs/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='a'
)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sample-k", type=int, required=True)
    args.add_argument("--confidence", type=float, required=True)
    args.add_argument("--mask", type=str, required=True)
    args.add_argument("--result", type=str, required=True)
    args.add_argument("--climate", type=str, required=True)
    args.add_argument("--sample", type=str, required=True)
    args = args.parse_args()
    if args.sample_k is None:
        raise ValueError("sample_k is required")
    if args.confidence is None:
        raise ValueError("confidence is required")
    if args.mask is None:
        raise ValueError("mask is required")
    if args.result is None:
        raise ValueError("result is required")
    if args.climate is None:
        raise ValueError("climate is required")
    if args.sample is None:
        raise ValueError("sample is required")
    main(
        sample_k=args.sample_k,
        confidence=args.confidence,
        mask_file_path=args.mask,
        output_dir=args.result,
        climate_dir=args.climate,
        sample_file=args.sample
    )
