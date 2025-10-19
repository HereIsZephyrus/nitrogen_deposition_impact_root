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
    standardized_raster = standardizer.standardize(stacked_raster.data)

    use_mpi_for_calc = MPI_AVAILABLE and (comm is not None) and (comm.Get_size() > 1)
    global_gmm_params = {
        "confidence": confidence,
        "max_iter": 100,
        "tol": 1e-6,
        "random_state": 42,
        "n_init": 3,
        "use_mpi": use_mpi_for_calc  # Use MPI for log likelihood calculation inside GMMCluster
    }
    besk_k = 14 # has already been selected
    #besk_k = select_optimal_k_with_aic(
    #    X=res_data,
    #    k_range=range(9, 12),
    #    **global_gmm_params
    #)
    #if besk_k is None:
    #    if rank == 0:
    #        logger.error("No optimal k found")
    #    raise ValueError("No optimal k found")
    # All processes create global_cluster and participate in fitting
    global_cluster = GMMCluster(
        n_components=besk_k,
        **global_gmm_params
    )
    valid_mask = np.all(np.isfinite(standardized_raster), axis=-1)
    valid_raster = standardized_raster[valid_mask]

    if rank == 0:
        logger.info(f"Fitting global GMM with {besk_k} components")
    global_cluster.fit(valid_raster)

    # All processes predict, but only rank 0 saves sample results
    sample_cluster_result = global_cluster.predict_with_confidence(standardized_sample)
    if rank == 0:
        sample_cluster_result.save(os.path.join(output_dir, 'sample_cluster_result.csv'))

    # All processes predict cluster results
    cluster_result = global_cluster.predict(valid_raster)
    full_total_labels = np.full(valid_mask.shape, -1, dtype=np.int32)  # -1 for NoData
    valid_total_labels = cluster_result.labels.copy() + 1  # Start from 1
    full_total_labels[valid_mask] = valid_total_labels

    # Only rank 0 writes final results
    if rank == 0:
        cluster_result_path = os.path.join(output_dir, 'total_cluster_result.tif')
        write_raster(
            data = full_total_labels,
            ref_raster = stacked_raster,
            output_path = cluster_result_path
        )
        cluster_result.save(os.path.join(output_dir, 'cluster_result.csv'))
