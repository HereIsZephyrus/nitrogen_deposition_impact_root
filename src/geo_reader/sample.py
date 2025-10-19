"""
Raster sampling utilities using GDAL
"""
import os
import logging
from typing import List, Tuple, Union
from math import floor
import numpy as np
import pandas as pd
from osgeo import gdal
from .raster_reader import Raster
from .sample_reader import Sample
from .raster_reader import RasterReader
from ..variance import Variance, ClimateVariance, SoilVariance, VegetationVariance, NitrogenVariance

logger = logging.getLogger(__name__)

def sample_raster(raster: Raster, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Sample raster data at given points using GDAL

    Args:
        raster: Raster object to sample from
        points: List of (x, y) coordinates. 

    Returns:
        Array of sampled values with shape (n_points, bands) or (n_points,) for single band
    """
    if not points:
        return None

    if raster.geotransform is None:
        logger.error("Raster must have valid geotransform")
        return None

    # Get inverse geotransform for coordinate conversion
    gt_forward = raster.geotransform
    gt_reverse = gdal.InvGeoTransform(gt_forward)
    if gt_reverse is None:
        logger.error("Cannot compute inverse geotransform")
        return None

    n_bands = raster.data.shape[0]
    n_points = len(points)
    sampled_values = np.full((n_points, n_bands), np.nan, dtype=np.float32)

    px_list = []
    py_list = []
    for (mx, my) in points:
        px, py = gdal.ApplyGeoTransform(gt_reverse, mx, my)
        px = int(floor(px))  # x pixel (column)
        py = int(floor(py))  # y pixel (row)
        px_list.append(px)
        py_list.append(py)

    sampled_values = raster.data[:, py_list, px_list]
    return sampled_values

def get_sample_data(sample_file: str, raster: Raster, grouped: bool = True) -> np.ndarray:
    """
    Sample climate data from raster based on sample locations

    Args:
        sample_file: Path to sample CSV file
        raster: Climate raster data
        grouped: Whether to deduplicate by group (default True)

    Returns:
        Array with shape (n_bands, n_samples) where n_samples depends on grouped parameter
    """
    samples = Sample(sample_file, grouped=grouped)
    sample_locations = samples.get_location()
    # sample in raster
    sample_data = sample_raster(raster, sample_locations)

    return sample_data

def stack_rasters(raster_list: List[Raster]) -> Raster:
    """
    Stack rasters into a single raster

    Args:
        raster_list: List of Raster objects

    Returns:
        Raster object with stacked data
    """
    stacked_data = np.stack([raster.data for raster in raster_list], axis=0)
    return Raster(stacked_data, raster_list[0].geotransform, raster_list[0].resolution)

def load_variance(climate_dir: str, sample: Sample) -> Union[List[Variance], Tuple[List[Variance], List[int]]]:
    """
    Load variance from sample file and climate raster data

    Args:
        climate_dir: Directory containing climate raster files (bio1-bio19, elevation)
        sample: Sample object

    Returns:
        If sample.grouped is True: List of Variance objects (one per unique group)
        If sample.grouped is False: Tuple of (List of Variance objects, List of group IDs)
    """

    # Load and stack climate rasters
    raster_list = [
        RasterReader.read_file(file_path=os.path.join(climate_dir, tif_file)) for tif_file in sorted(os.listdir(climate_dir))
    ]
    raster_list = [raster for raster in raster_list if raster is not None]

    if not raster_list:
        logger.error("No valid rasters found in %s", climate_dir)
        if sample.grouped:
            return []
        else:
            return [], []

    stacked_raster = stack_rasters(raster_list)

    # Sample climate data from rasters
    climate_data = get_sample_data(sample.sample_file, stacked_raster, grouped=sample.grouped)

    # Get data from sample using various get methods
    # Behavior depends on sample.grouped attribute
    soil_data = sample.get_soil()
    nitrogen_data = sample.get_nitrogen()
    vegetation_data = sample.get_vegetation()

    # Get group information for non-grouped mode
    group_list = None
    if not sample.grouped:
        group_list = sample.get_group()

    # Determine processing mode based on sample.grouped attribute
    if sample.grouped:
        # Grouped mode: data is deduplicated by group
        n_samples = climate_data.shape[1] if climate_data.ndim > 1 else len(climate_data)
        n_total_records = len(sample)  # Total records in CSV (before deduplication)
        logger.info("Processing %d unique groups from %d total samples (grouped mode)", n_samples, n_total_records)
    else:
        # Non-grouped mode: process all individual records
        n_samples = len(sample)  # All individual records
        logger.info("Processing %d individual samples (non-grouped mode)", n_samples)

    # Build variance list
    variance_list = []

    for i in range(n_samples):
        try:
            # Extract climate data for this sample
            # Assuming climate_data has shape (n_bands, n_samples) with bands ordered as:
            # bio1, bio12, bio2-bio11, bio13-bio19, elevation
            climate_values = climate_data[:, i]

            # Map climate values to ClimateVariance fields
            # Expected order: bio1, bio12, bio2-bio11, bio13-bio19, elevation (20 values)
            if len(climate_values) >= 20:
                climate_var = ClimateVariance(
                    bio1=float(climate_values[0]),
                    bio12=float(climate_values[1]),
                    bio2=float(climate_values[2]),
                    bio3=float(climate_values[3]),
                    bio4=float(climate_values[4]),
                    bio5=float(climate_values[5]),
                    bio6=float(climate_values[6]),
                    bio7=float(climate_values[7]),
                    bio8=float(climate_values[8]),
                    bio9=float(climate_values[9]),
                    bio10=float(climate_values[10]),
                    bio11=float(climate_values[11]),
                    bio13=float(climate_values[12]),
                    bio14=float(climate_values[13]),
                    bio15=float(climate_values[14]),
                    bio16=float(climate_values[15]),
                    bio17=float(climate_values[16]),
                    bio18=float(climate_values[17]),
                    bio19=float(climate_values[18]),
                    elevation=float(climate_values[19])
                )
            else:
                logger.warning("Insufficient climate data for sample %d, skipping", i)
                continue

            # Extract soil data for this sample (20 features)
            if i < len(soil_data) and len(soil_data[i]) >= 20:
                soil_values = soil_data[i]
                soil_var = SoilVariance(
                    available_water_capacity_for_rootable_soil_depth=float(soil_values[0]),
                    coarse_fragments=float(soil_values[1]),
                    sand=float(soil_values[2]),
                    silt=float(soil_values[3]),
                    clay=float(soil_values[4]),
                    bulk=float(soil_values[5]),
                    organic_carbon_content=float(soil_values[6]),
                    ph_in_water=float(soil_values[7]),
                    total_nitrogen_content=float(soil_values[8]),
                    cn_ratio=float(soil_values[9]),
                    cec_soil=float(soil_values[10]),
                    cec_clay=float(soil_values[11]),
                    cec_eff=float(soil_values[12]),
                    teb=float(soil_values[13]),
                    bsat=float(soil_values[14]),
                    alum_sat=float(soil_values[15]),
                    esp=float(soil_values[16]),
                    tcarbon_eq=float(soil_values[17]),
                    gypsum=float(soil_values[18]),
                    elec_cond=float(soil_values[19])
                )
            else:
                logger.warning("Insufficient soil data for sample %d, skipping", i)
                continue

            # Extract vegetation data for this sample
            if i < len(vegetation_data):
                vegetation_var = VegetationVariance(
                    vegetation_type=int(vegetation_data[i])
                )
            else:
                logger.warning("Missing vegetation data for sample %d, skipping", i)
                continue

            # Extract nitrogen data for this sample (4 features)
            if i < len(nitrogen_data) and len(nitrogen_data[i]) >= 4:
                nitrogen_values = nitrogen_data[i]
                nitrogen_var = NitrogenVariance(
                    n_addition=float(nitrogen_values[0]),
                    fertilizer_type=int(nitrogen_values[1]),
                    treatment_date=int(nitrogen_values[2]),
                    duration=float(nitrogen_values[3])
                )
            else:
                logger.warning("Insufficient nitrogen data for sample %d, skipping", i)
                continue

            # Create complete Variance object
            variance = Variance(
                climate=climate_var,
                soil=soil_var,
                vegetation=vegetation_var,
                nitrogen=nitrogen_var
            )

            variance_list.append(variance)

        except (ValueError, IndexError, TypeError) as e:
            logger.warning("Error creating Variance for group %d: %s", i, e)
            continue

    if sample.grouped:
        logger.info("Created %d Variance objects from %d unique groups (%d total records)", 
                    len(variance_list), n_samples, len(sample))
        return variance_list
    else:
        logger.info("Created %d Variance objects from %d individual samples", 
                    len(variance_list), n_samples)
        return variance_list, group_list

def get_climate_group(sample_cluster_result_file: str) -> np.ndarray:
    """
    Get climate group from sample cluster result file

    Args:
        sample_cluster_result_file: Path to sample cluster result file

    Returns:
        Array of climate group
    """
    sample_cluster_result = pd.read_csv(sample_cluster_result_file)
    return sample_cluster_result['labels'].values.astype(int)
