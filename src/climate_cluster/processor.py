import os
import logging
from typing import List
import numpy as np
from .standardize import Standardizer
from .cluster import GMMCluster, select_optimal_k_with_aic
from ..geo_reader import RasterReader, get_sample_data, Raster
from ..variance import ClimateVariance

logger = logging.getLogger(__name__)

def stack_rasters(raster_list: List[Raster]) -> Raster:
    """
    Stack rasters into a single raster
    """
    stacked_data = np.concatenate([raster.data for raster in raster_list], axis=0)
    return Raster(stacked_data, raster_list[0].geotransform, raster_list[0].resolution)

def main(sample_k: int, confidence: float, output_dir: str, climate_dir: str, sample_file: str):
    """
    Main function to process climate clustering

    Args:
        sample_k: Number of clusters for sample data
        confidence: Confidence threshold for classification
        output_dir: Path to output directory
        climate_dir: Path to climate data directory
        sample_file: Path to sample file
    """
    dim = len(ClimateVariance.model_fields)
    raster_list = [RasterReader.read_file(file_path=os.path.join(climate_dir, tif_file), resolution=0.5) for tif_file in os.listdir(climate_dir)]
    stacked_raster = stack_rasters(raster_list)
    sample_data = get_sample_data(sample_file, stacked_raster)

    standardizer = Standardizer(dimension=dim)
    standardizer.fit(stacked_raster)
    standardized_sample = standardizer.standardize(sample_data)
    standardized_raster = standardizer.standardize(stacked_raster.data)

    sample_cluster = GMMCluster(
        n_components=sample_k,
        confidence=confidence,
        max_iter=100,
        tol=1e-6,
        random_state=42,
        n_init=10
    )
    sample_cluster.fit(standardized_sample)
    sample_cluster_result = sample_cluster.predict_with_confidence(standardized_sample)
    raster_cluster_first_result = sample_cluster.predict_with_confidence(standardized_raster)
    sample_cluster_result.save(output_dir)
    raster_cluster_first_result.save(output_dir)
    res_data = np.concatenate([
        sample_cluster_result.exclude(standardized_sample),
        raster_cluster_first_result.exclude(standardized_raster)],
        axis=0)
    besk_k = select_optimal_k_with_aic(res_data)
    if besk_k is None:
        logger.error("No optimal k found")
        raise ValueError("No optimal k found")
    res_cluster = GMMCluster(
        n_components=besk_k,
        confidence=confidence,
        max_iter=100,
        tol=1e-6,
        random_state=42,
        n_init=10
    )
    res_cluster.fit(res_data)
    res_cluster_result = res_cluster.predict(res_data)
    res_cluster_result.save(output_dir)
