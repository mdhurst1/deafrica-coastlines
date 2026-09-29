#!/usr/bin/env python
"""Generate annual DEA-style MSL shorelines for every polygon in an AOI file.

The script deliberately only orchestrates the existing Coastlines functions:
AOI -> DEM -> site-specific tidal cutoffs -> filtered Landsat -> annual/gapfill
rasters -> DEA preprocessing -> sub-pixel MSL shorelines.

It is resumable: existing DEMs, cutoffs, yearly observation/tide caches,
annual rasters and final shoreline files are reused automatically.

This script and workflow was developed with support by ChatGPT

MDH, September 2026

"""

# import modules
from __future__ import annotations
import re
import time
from pathlib import Path

# general data tools
import geopandas as gpd
import numpy as np
import planetary_computer.sas as pc_sas
import rioxarray
import xarray as xr
from shapely.geometry import Point

# dea specific tools
from dea_tools.spatial import subpixel_contours
from eo_tides.eo import pixel_tides
from coastlines.raster import load_tidal_subset, tidal_composite, tide_cutoffs
from coastlines.stac import load_dem_stac, load_water_index_stac
from coastlines.vector import contour_certainty, contours_preprocess, load_rasters

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
# All paths are defined here so the script can be run directly with:
#
#     python scripts/run_scotland_shorelines.py
#
# Set SITES to None to process every AOI, or provide one or more exact names
# from the GeoJSON, e.g. ["Montrose Bay"], while testing.
RESULTS_FOLDER = Path("/home/mh322u/dea_scotland_results")
AOI_FILE = RESULTS_FOLDER / "UKRI_Policy_Fellowship_Planet_AOIs_Exemplar_Sites.geojson"
TIDE_MODEL_DIR = Path("/home/mh322u/tide_models")
NAME_FIELD = "Name"
SITES = None

# -----------------------------------------------------------------------------
# Processing settings
# -----------------------------------------------------------------------------
CRS = "EPSG:27700"  # British National Grid
RESOLUTION = 30
START_YEAR, END_YEAR = 1988, 2025
CUTOFF_START, CUTOFF_END = 1984, 2025
TIDE_MODEL = "EOT20"
PLATFORMS = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
CLOUD_COVER = 80
SHADOW_THRESHOLD, SHADOW_RADIUS = 0.5, 1
WATER_INDEX, INDEX_THRESHOLD = "mndwi", 0.0
RASTER_ROOT = RESULTS_FOLDER / "outputs" / "dea_rasters"
SITE_ROOT = RESULTS_FOLDER / "outputs" / "aoi_analysis"


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def rasters_exist(folder, year, suffix=""):
    return all(
        (folder / f"{year}_{v}{suffix}.tif").exists()
        for v in ["mndwi", "ndwi", "count", "stdev", "tide_m"]
    )


def ocean_seeds(yearly_ds, max_points=50):
    """Use persistently wet raster-edge pixels as ocean seed points."""
    index = yearly_ds[WATER_INDEX]
    wet = ((index >= INDEX_THRESHOLD).where(index.notnull())).mean("year").compute()
    valid = index.notnull().mean("year").compute()
    edge = np.zeros(wet.shape, dtype=bool)
    edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = True
    candidates = np.argwhere(edge & (valid.values >= 0.25) & (wet.values >= 0.70))
    if len(candidates) == 0:
        raise RuntimeError("No persistent-water pixels found on the raster edge")
    if len(candidates) > max_points:
        candidates = candidates[np.linspace(0, len(candidates) - 1, max_points).astype(int)]
    points = [
        Point(float(yearly_ds.x.values[x]), float(yearly_ds.y.values[y]))
        for y, x in candidates
    ]
    return gpd.GeoDataFrame(geometry=points, crs=yearly_ds.rio.crs)


    # Load, shadow-mask and tide-filter one year, caching the downloaded pixels.
    def observations(year):
        obs_file, tide_file = obs_dir / f"{year}_tidal.nc", tide_dir / f"{year}_tides.nc"
        if obs_file.exists():
            with xr.open_dataset(obs_file) as ds:
                return ds.load()

        if tide_file.exists():
            with xr.open_dataset(tide_file) as ds:
                tides = ds.tide_m.load()
        else:
            tides = pixel_tides(
                data=load_year(year), model=TIDE_MODEL,
                directory=TIDE_MODEL_DIR, resample=True,
            ).rename("tide_m").compute()
            tides.to_dataset().to_netcdf(tide_file)

        for attempt in range(1, 4):
            pc_sas.TOKEN_CACHE.clear()
            try:
                ds = load_year(year, shadow=True, fail_on_error=(attempt < 3))
                if not set(ds.time.values).issubset(set(tides.time.values)):
                    tides = pixel_tides(
                        data=ds, model=TIDE_MODEL, directory=TIDE_MODEL_DIR, resample=True
                    ).rename("tide_m").compute()
                    tides.to_dataset().to_netcdf(tide_file)
                ds = ds[["mndwi", "ndwi"]]
                ds["tide_m"] = tides.sel(time=ds.time)
                like = ds.tide_m.isel(time=0)
                filtered = load_tidal_subset(
                    ds, cutoff_min.interp_like(like), cutoff_max.interp_like(like)
                )
                filtered.to_netcdf(obs_file)
                return filtered
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5)

    # STAGE 4 -----------------------------------------------------------------
    # DEA annual median and true three-year gapfill composites.
    print("Creating/reusing annual raster composites...")
    previous, current, future = (
        observations(START_YEAR - 1), observations(START_YEAR), observations(START_YEAR + 1)
    )
    for year in range(START_YEAR, END_YEAR + 1):
        if not rasters_exist(raster_dir, year):
            tidal_composite(current, year, "year", str(raster_dir), export_geotiff=True)
        if not rasters_exist(raster_dir, year, "_gapfill"):
            gapfill = xr.concat([previous, current, future], dim="time").sortby("time")
            tidal_composite(
                gapfill, year, "year", str(raster_dir),
                output_suffix="_gapfill", export_geotiff=True,
            )
        if year < END_YEAR:
            previous, current, future = current, future, observations(year + 2)

    # STAGE 5 -----------------------------------------------------------------
    # Existing DEA coastal masking, sub-pixel contouring and certainty attribution.
    print("Extracting annual MSL shorelines...")
    yearly_ds, gapfill_ds = load_rasters(
        path=str(raster_dir), water_index=WATER_INDEX,
        start_year=START_YEAR, end_year=END_YEAR,
    )
    seeds = ocean_seeds(yearly_ds)
    seeds.to_file(vector_dir / "ocean_seed_points.gpkg", driver="GPKG")
    masked_ds, certainty = contours_preprocess(
        yearly_ds=yearly_ds, gapfill_ds=gapfill_ds, water_index=WATER_INDEX,
        index_threshold=INDEX_THRESHOLD, tide_points_gdf=seeds, buffer_pixels=33,
        mask_landcover=False, mask_ndwi=True, mask_temporal=True,
        mask_modifications=None,
    )
    contours = subpixel_contours(
        da=masked_ds, z_values=INDEX_THRESHOLD, min_vertices=10, dim="year"
    ).set_index("year")
    contours.index = contours.index.astype(str)
    contours = contour_certainty(contours, {str(y): m for y, m in certainty.items()})
    contours.index = contours.index.astype(int)
    contours.index.name = "year"

    shorelines = contours.reset_index()
    shorelines = shorelines[shorelines.geometry.notnull() & ~shorelines.geometry.is_empty].copy()
    shorelines = shorelines.to_crs(CRS)
    shorelines["geometry"] = shorelines.geometry.intersection(geometry)
    shorelines = shorelines[~shorelines.geometry.is_empty].copy()
    shorelines["indicator"], shorelines["tide_model"] = "DEA_MSL", TIDE_MODEL
    shorelines["segment_no"] = shorelines.groupby("year").cumcount() + 1
    shorelines["shoreline_id"] = (
        shorelines.year.astype(str) + "_" + shorelines.segment_no.astype(str).str.zfill(3)
    )
    shorelines.to_file(vector_dir / f"{site}_DEA_MSL_{START_YEAR}_{END_YEAR}_BNG.gpkg", driver="GPKG")
    shorelines.to_crs("EPSG:4326").to_file(output, driver="GeoJSON")
    print(f"Finished {name}: {output}")


def main():
    """Load the AOI file and run the shoreline workflow for each selected site."""
    if not AOI_FILE.exists():
        raise FileNotFoundError(f"AOI file not found: {AOI_FILE}")
    if not TIDE_MODEL_DIR.exists():
        raise FileNotFoundError(f"Tide model directory not found: {TIDE_MODEL_DIR}")

    aois = gpd.read_file(AOI_FILE)
    if aois.crs is None:
        raise ValueError("AOI file has no CRS")
    if NAME_FIELD not in aois.columns:
        raise ValueError(f"AOI file has no '{NAME_FIELD}' field")

    # Use British National Grid for all raster/vector processing.
    aois = aois.to_crs(CRS)

    # Optional testing subset. Leave SITES = None to process all AOIs.
    if SITES is not None:
        wanted = {site.casefold() for site in SITES}
        aois = aois[
            aois[NAME_FIELD].astype(str).str.casefold().isin(wanted)
        ]

    print(f"Processing {len(aois)} AOI(s) from {AOI_FILE.name}")

    # Each AOI is independent. If one fails, report it and continue to the next.
    for _, row in aois.iterrows():
        try:
            run_aoi(row)
        except Exception as error:
            print(f"ERROR: {row[NAME_FIELD]} failed: {error}")
