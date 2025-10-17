import numpy as np
from typing import List, Dict, Any
from .standardize import Standardizer
from .cluster import GMMCluster, select_optimal_k_with_aic
from ..geo_reader import Raster, sample_raster
from ..variance import ClimateVariance

def calc_cluster(sample_k: int, confidence: float, output_dir: str):
    pass

def main(sample_k: int, confidence: float, mask_file_path: str, output_dir: str, climate_dir: str, sample_file: str):
    dim = len(ClimateVariance.__fields__)
    standardizer = Standardizer(dimension=dim, land_mask_path=mask_file_path)

    standardizer.fit(data)
    cluster = GMMCluster(n_components=sample_k, max_iter=100, tol=1e-6, random_state=42, n_init=10)
    cluster.fit(standardizer.standardize(data))
    cluster.predict(standardizer.standardize(data))
    cluster.predict_with_confidence(standardizer.standardize(data))
    cluster.predict_with_confidence(standardizer.standardize(data))
