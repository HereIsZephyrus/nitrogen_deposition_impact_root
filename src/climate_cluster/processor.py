import os
import logging
from typing import List
import numpy as np
from .standardize import Standardizer
from .cluster import GMMCluster, select_optimal_k_with_aic
from ..geo_reader import RasterReader, get_sample_data, write_raster, stack_rasters
from ..variance import ClimateVariance

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

logger = logging.getLogger(__name__)

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
    # Determine if running in MPI environment
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        logger.info(f"Running with MPI: rank {rank} of {size}")
    else:
        rank = 0
        comm = None

    # Only rank 0 reads and preprocesses data
    if rank == 0:
        dim = len(ClimateVariance.model_fields)
        raster_list = [RasterReader.read_file(file_path=os.path.join(climate_dir, tif_file), resolution=0.5) for tif_file in os.listdir(climate_dir)]
        raster_list = [raster for raster in raster_list if raster is not None]
        stacked_raster = stack_rasters(raster_list)
        sample_data = get_sample_data(sample_file, stacked_raster)
    else:
        stacked_raster = None
        sample_data = None
        dim = None

    # Broadcast data to all processes if using MPI
    if MPI_AVAILABLE:
        stacked_raster = comm.bcast(stacked_raster, root=0)
        sample_data = comm.bcast(sample_data, root=0)
        dim = comm.bcast(dim, root=0)

    standardizer = Standardizer(dimension=dim)
    standardizer.fit(stacked_raster)
    standardized_sample = standardizer.standardize(sample_data)

    sample_cluster = GMMCluster(
        n_components=sample_k,
        confidence=confidence,
        max_iter=100,
        tol=1e-6,
        random_state=42,
        n_init=50,
        max_covariance_det=8.0,
        min_cluster_separation=0.4,
        max_mean_mahalanobis=4.0
    )
    sample_cluster.fit(standardized_sample)
    sample_cluster_result = sample_cluster.predict_with_confidence(standardized_sample)

    # Only rank 0 saves results
    if rank == 0:
        sample_cluster_result.save(os.path.join(output_dir, 'sample_cluster_result.csv'))

    standardized_raster = standardizer.standardize(stacked_raster.data)

    valid_mask = np.all(np.isfinite(standardized_raster), axis=-1)
    if rank == 0:
        logger.info(f"Valid raster pixels: {np.sum(valid_mask)} / {valid_mask.size} ({100*np.sum(valid_mask)/valid_mask.size:.2f}%)")

    valid_raster_data = standardized_raster[valid_mask]
    raster_cluster_first_result = sample_cluster.predict_with_confidence(valid_raster_data)

    full_labels = np.full(valid_mask.shape, -1, dtype=np.int32)  # -1 for NoData

    valid_labels = np.full(valid_raster_data.shape[0], 0, dtype=np.int32)
    valid_labels[raster_cluster_first_result.confident_mask] = raster_cluster_first_result.labels[raster_cluster_first_result.confident_mask] + 1
    full_labels[valid_mask] = valid_labels

    # Only rank 0 writes raster
    if rank == 0:
        first_raster_cluster_result_path = os.path.join(output_dir, 'first_raster_cluster_result.tif')
        write_raster(
            data = full_labels,
            ref_raster = stacked_raster,
            output_path = first_raster_cluster_result_path
        )

    # Exclude confident pixels from further clustering
    res_data = raster_cluster_first_result.exclude(valid_raster_data)

    # Determine if we should use MPI for log likelihood calculation
    # Only use MPI if we're in an MPI environment (multiple processes)
    use_mpi_for_calc = MPI_AVAILABLE and (comm is not None) and (comm.Get_size() > 1)

    global_gmm_params = {
        "confidence": confidence,
        "max_iter": 100,
        "tol": 1e-6,
        "random_state": 42,
        "n_init": 3,
        "use_mpi": use_mpi_for_calc  # Use MPI for log likelihood calculation inside GMMCluster
    }
    besk_k = 11 # has already been selected
    #besk_k = select_optimal_k_with_aic(
    #    X=res_data,
    #    k_range=range(9, 12),
    #    **global_gmm_params
    #)
    #if besk_k is None:
    #    if rank == 0:
    #        logger.error("No optimal k found")
    #    raise ValueError("No optimal k found")
    res_cluster = GMMCluster(
        n_components=besk_k,
        **global_gmm_params
    )
    res_cluster.fit(res_data)
    res_cluster_result = res_cluster.predict(res_data)

    # Reconstruct full raster with both first and second clustering results
    full_total_labels = np.full(valid_mask.shape, -1, dtype=np.int32)  # -1 for NoData

    valid_total_labels = raster_cluster_first_result.labels.copy() + 1  # Start from 1
    not_confident_mask = ~raster_cluster_first_result.confident_mask
    valid_total_labels[not_confident_mask] = res_cluster_result.labels + sample_k + 1

    full_total_labels[valid_mask] = valid_total_labels

    # Only rank 0 writes final results
    if rank == 0:
        res_cluster_result_path = os.path.join(output_dir, 'total_cluster_result.tif')
        write_raster(
            data = full_total_labels,
            ref_raster = stacked_raster,
            output_path = res_cluster_result_path
        )
        res_cluster_result.save(os.path.join(output_dir, 'second_cluster_result.csv'))
