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

import sys
from pathlib import Path
from typing import List
from .processor import run_complete_climate_clustering
