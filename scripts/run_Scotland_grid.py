#!/usr/bin/env python
"""Run DEA-style annual MSL shoreline extraction on a national 20 km BNG grid.

This is a separate grid-based driver for scaling the Scotland workflow beyond
the exemplar AOIs.

Each grid tile has two spatial extents:

    CORE
        The authoritative 20 x 20 km BNG tile. Final shoreline vectors are
        clipped to this geometry so every national output location belongs to
        exactly one tile.

    PROCESSING FOOTPRINT
        A 5 km buffer around the core, implemented as a 30 x 30 km BNG box.
        DEM, Landsat, tides, annual composites and shoreline extraction are
        performed across this larger area to minimise tile-edge effects.

The driver uses the same versioned raster/vector structure as the successful
exemplar workflow and is resumable. The default selection is the 11-tile
regional pilot. Use --all later to process every coastal tile.

Examples
--------
Pilot, resumable:
    python -u scripts/run_scotland_grid_shorelines.py

Pilot, rebuild current version from scratch:
    python -u scripts/run_scotland_grid_shorelines.py --fresh

Debug one or more tiles:
    python -u scripts/run_scotland_grid_shorelines.py --tiles 17 18 --fresh

Full national grid:
    python -u scripts/run_scotland_grid_shorelines.py --all

Pilot extraction + DEA change analysis + merged products:
    python -u scripts/run_scotland_grid_shorelines.py --analyse

National analysis after tile extraction is complete:
    python -u scripts/run_scotland_grid_shorelines.py --all --analyse

For parallel national extraction, run workers WITHOUT --analyse, then run a
single --all --analyse pass afterwards to create the merged national products.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import odc.algo
import pandas as pd
import planetary_computer.sas as pc_sas
import rioxarray
import xarray as xr
from dea_tools.spatial import subpixel_contours
from eo_tides.eo import pixel_tides
from shapely.geometry import Point, box
from shapely.ops import unary_union
from skimage.morphology import binary_closing, disk

from coastlines.raster import load_tidal_subset, tidal_composite, tide_cutoffs
from coastlines.stac import load_dem_stac, load_water_index_stac
from coastlines.vector import (
    all_time_stats,
    annual_movements,
    calculate_regressions,
    coastal_masking,
    contour_certainty,
    contours_preprocess,
    load_rasters,
    ocean_masking,
    points_on_line,
    temporal_masking,
)


# =============================================================================
# USER SETTINGS
# =============================================================================

# folders
HOME_FOLDER = Path("/media/nevis_array/user_homes/mhurst/")
RESULTS_FOLDER = HOME_FOLDER / "scotland_dea_results"
TIDE_MODEL_DIR = HOME_FOLDER / "tide_models"

# Copy the 20 km core grid into the repository data folder.
GRID_FILE = HOME_FOLDER / "Scotland_DEA_20km_core_tiles.geojson"
TILE_ID_FIELD = "tile_id"
TILE_NO_FIELD = "tile_no"

# Default regional upscale pilot. Set to None to make all tiles the default,
# or keep this list and use --all when ready for the national run.
PILOT_TILE_NOS = [17, 18, 22, 23, 28, 34, 35, 43, 44, 45, 46]

# Processing context around every 20 km core tile.
PROCESSING_BUFFER_M = 5000


# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

CRS = "EPSG:27700"
WGS84 = "EPSG:4326"
RESOLUTION = 30

START_YEAR, END_YEAR = 1988, 2025
CUTOFF_START, CUTOFF_END = 1984, 2025

TIDE_MODEL = "FES2022_extrapolated"
PLATFORMS = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
CLOUD_COVER = 80

SHADOW_THRESHOLD = 0.5
SHADOW_RADIUS = 1

WATER_INDEX = "mndwi"
INDEX_THRESHOLD = 0.0

# Scotland default. A 5-pixel (~150 m) morphological closing retains narrow
# marine/estuarine connections better than the original DEA default of 15.
COASTAL_CLOSING_DEFAULT = 5

# Keep this dictionary for genuine local exceptions discovered during QA.
COASTAL_CLOSING_OVERRIDES = {
}

# Pilot QA: export diagnostic masks without changing the shoreline method.
EXPORT_MASK_DIAGNOSTICS = True
DIAGNOSTIC_YEAR = 2020


##
# ANALYTICAL SETTINGS
##

BASELINE_YEAR = 2024
POINT_SPACING_M = 30
ANALYSIS_CONTEXT_M = 5000
MAX_VALID_DIST_M = 5000

# =============================================================================
# VERSIONING
# =============================================================================

# Keep this pilot distinct from the earlier AOI versions. If the regional pilot
# leads to methodological changes, increment this before the national run.
RASTER_VERSION = "scotland_grid_v0.1_bng_shadow"
VECTOR_VERSION = "scotland_grid_v0.2_closing5"


# =============================================================================
# OUTPUT ROOTS
# =============================================================================

# Change RESULTS_ROOT to a mounted large drive before national production if
# desired, e.g. Path("/path/to/large_drive/dea_scotland").
RASTER_ROOT = RESULTS_FOLDER / "dea_rasters"
VECTOR_ROOT = RESULTS_FOLDER / "dea_vectors"
CACHE_ROOT = RESULTS_FOLDER / "dea_cache"
LOG_ROOT = RESULTS_FOLDER / "run_logs"
ANALYSIS_ROOT = RESULTS_FOLDER / "dea_analysis"


# =============================================================================
# ARGUMENTS
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DEA-style Scotland shoreline extraction on 20 km grid tiles."
    )

    selection = parser.add_mutually_exclusive_group()

    selection.add_argument(
        "--all",
        action="store_true",
        help="Process every tile in GRID_FILE rather than the default pilot.",
    )

    selection.add_argument(
        "--tiles",
        nargs="+",
        type=int,
        help="Process specific tile_no values, e.g. --tiles 17 18 22.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete cache, raster and vector products for the selected tiles "
            "for the CURRENT versions before rerunning."
        ),
    )

    parser.add_argument(
        "--analyse",
        action="store_true",
        help=(
            "Generate/reuse per-tile DEA baseline-point change analysis, then "
            "merge completed shoreline and rate outputs for the current selection. "
            "For national parallel processing, use disjoint --tiles selections "
            "and finish with one --all --analyse pass."
        ),
    )

    return parser.parse_args()


# =============================================================================
# NETCDF / FILE HELPERS
# =============================================================================


def save_netcdf_safe(obj, path):
    """Safely write xarray data while handling object-dtype metadata.

    STAC/ODC/eo-tides can attach scalar auxiliary coordinates such as
    tide_model="EOT20" using Python object dtype. NetCDF4 cannot serialise these
    coordinates directly. Scalar metadata are moved to global attributes;
    other non-dimensional object coordinates are dropped. Scientific object-
    dtype data variables are never silently discarded.
    """
    if isinstance(obj, xr.DataArray):
        name = obj.name or "data"
        ds = obj.rename(name).to_dataset()
    else:
        ds = obj.copy()

    for name in list(ds.coords):
        var = ds.coords[name]

        if var.dtype.kind != "O":
            continue

        if name in ds.dims:
            # Dimension coordinates must be retained.
            try:
                ds = ds.assign_coords({name: var.astype(str)})
            except Exception as exc:
                raise TypeError(
                    f"Cannot serialise object-dtype dimension coordinate {name!r}"
                ) from exc

        elif var.size == 1:
            # Preserve scalar metadata such as tide_model="EOT20".
            value = np.asarray(var.values).reshape(-1)[0]
            ds.attrs[name] = str(value)
            print(
                f"Moving auxiliary coordinate {name!r}={value!r} "
                "to NetCDF attributes"
            )
            ds = ds.drop_vars(name)

        else:
            print(
                f"Dropping non-NetCDF auxiliary coordinate {name!r} "
                "(dtype=object)"
            )
            ds = ds.drop_vars(name)

    bad_vars = [
        name
        for name, var in ds.data_vars.items()
        if var.dtype.kind == "O"
    ]
    if bad_vars:
        raise TypeError(
            "Cannot write object-dtype data variable(s) to NetCDF: "
            + ", ".join(bad_vars)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    try:
        ds.to_netcdf(tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def rasters_exist(folder, year, suffix=""):
    variables = ["mndwi", "ndwi", "count", "stdev", "tide_m"]
    return all(
        (folder / f"{year}_{variable}{suffix}.tif").exists()
        for variable in variables
    )


def directory_size_gib(path):
    """Return total file size below path in GiB."""
    path = Path(path)
    if not path.exists():
        return 0.0

    total = sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )
    return total / (1024 ** 3)


# =============================================================================
# GRID HELPERS
# =============================================================================


def validate_and_select_grid(args):
    """Load the national core grid and select pilot/requested tiles."""
    if not GRID_FILE.exists():
        raise FileNotFoundError(
            f"Grid file not found: {GRID_FILE}\n"
            "Copy Scotland_DEA_20km_core_tiles.geojson into the repo data folder."
        )

    grid = gpd.read_file(GRID_FILE)

    if grid.crs is None:
        raise ValueError("Grid file has no CRS.")

    grid = grid.to_crs(CRS)

    required = {TILE_ID_FIELD, TILE_NO_FIELD}
    missing_fields = required - set(grid.columns)
    if missing_fields:
        raise ValueError(
            f"Grid is missing required field(s): {sorted(missing_fields)}"
        )

    if grid[TILE_ID_FIELD].duplicated().any():
        raise ValueError("Grid contains duplicate tile_id values.")

    if grid[TILE_NO_FIELD].duplicated().any():
        raise ValueError("Grid contains duplicate tile_no values.")

    grid = grid[
        grid.geometry.notnull()
        & ~grid.geometry.is_empty
    ].copy()

    if args.all:
        selected = grid.copy()

    else:
        requested = args.tiles if args.tiles is not None else PILOT_TILE_NOS

        if requested is None:
            selected = grid.copy()
        else:
            selected = grid[
                grid[TILE_NO_FIELD].astype(int).isin(requested)
            ].copy()

            found = set(selected[TILE_NO_FIELD].astype(int))
            missing = sorted(set(requested) - found)

            if missing:
                raise ValueError(
                    f"Requested tile_no value(s) not found: {missing}"
                )

    selected = selected.sort_values(TILE_NO_FIELD).reset_index(drop=True)

    # Sanity check the regular 20 km core geometry.
    widths = selected.geometry.bounds.maxx - selected.geometry.bounds.minx
    heights = selected.geometry.bounds.maxy - selected.geometry.bounds.miny

    if not (
        np.allclose(widths, 20000, atol=1)
        and np.allclose(heights, 20000, atol=1)
    ):
        raise ValueError(
            "Selected grid contains cells that are not 20 x 20 km BNG cores."
        )

    return selected


def processing_footprint(core_geometry):
    """Return an exact rectangular 5 km context around a 20 km core."""
    minx, miny, maxx, maxy = core_geometry.bounds

    return box(
        minx - PROCESSING_BUFFER_M,
        miny - PROCESSING_BUFFER_M,
        maxx + PROCESSING_BUFFER_M,
        maxy + PROCESSING_BUFFER_M,
    )


def wgs84_bbox(geometry):
    """Convert one BNG geometry to a WGS84 bbox for STAC queries."""
    return list(
        gpd.GeoSeries([geometry], crs=CRS)
        .to_crs(WGS84)
        .total_bounds
    )


# =============================================================================
# VERSIONED PATHS
# =============================================================================


def versioned_paths(tile_id):
    """Return version-specific paths for one national grid tile."""
    cache = (
        CACHE_ROOT
        / RASTER_VERSION
        / f"{tile_id}_{RASTER_VERSION}"
    )

    raster_dir = (
        RASTER_ROOT
        / RASTER_VERSION
        / f"{tile_id}_{RASTER_VERSION}"
    )

    vector_dir = (
        VECTOR_ROOT
        / VECTOR_VERSION
        / f"{tile_id}_{VECTOR_VERSION}"
    )

    return cache, raster_dir, vector_dir


def clear_current_version(tile_id):
    """Delete only this tile's outputs for the current versions."""
    cache, raster_dir, vector_dir = versioned_paths(tile_id)

    for path in [cache, raster_dir, vector_dir]:
        if path.exists():
            print(f"Removing for fresh run: {path}")
            shutil.rmtree(path)


# =============================================================================
# OCEAN SEEDS
# =============================================================================


def ocean_seeds(yearly_ds, max_points=50):
    """Use persistently wet processing-edge pixels as ocean seed points."""
    index = yearly_ds[WATER_INDEX]

    wet = (
        (index >= INDEX_THRESHOLD)
        .where(index.notnull())
        .mean("year")
        .compute()
    )

    valid = index.notnull().mean("year").compute()

    edge = np.zeros(wet.shape, dtype=bool)
    edge[:3, :] = True
    edge[-3:, :] = True
    edge[:, :3] = True
    edge[:, -3:] = True

    candidates = np.argwhere(
        edge
        & (valid.values >= 0.25)
        & (wet.values >= 0.70)
    )

    if len(candidates) == 0:
        raise RuntimeError(
            "No persistent-water pixels found on the buffered processing edge"
        )

    if len(candidates) > max_points:
        candidates = candidates[
            np.linspace(
                0,
                len(candidates) - 1,
                max_points,
            ).astype(int)
        ]

    points = [
        Point(
            float(yearly_ds.x.values[x]),
            float(yearly_ds.y.values[y]),
        )
        for y, x in candidates
    ]

    return gpd.GeoDataFrame(
        geometry=points,
        crs=yearly_ds.rio.crs,
    )



# =============================================================================
# MASK DIAGNOSTICS
# =============================================================================


def write_mask_geotiff(mask, path, reference):
    """Write a 2-D diagnostic mask as a compact uint8 GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = mask.squeeze(drop=True)

    if out.dtype.kind == "b":
        out = out.astype("uint8")
    else:
        out = out.fillna(0).astype("uint8")

    # Remove inherited raster profile metadata so GDAL does not
    # interpret WIDTH, HEIGHT, CRS, TRANSFORM etc. as creation options.
    out = out.copy()
    out.attrs = {}
    out.encoding = {}

    # Re-attach only the spatial information we actually need.
    out = out.rio.write_crs(
        reference.rio.crs,
        inplace=False,
    )

    out.rio.write_transform(
        reference.rio.transform(),
        inplace=True,
    )

    out.rio.to_raster(
        path,
        driver="GTiff",
        dtype="uint8",
        compress="deflate",
    )


def export_mask_diagnostics(
    yearly_ds,
    gapfill_ds,
    tide_points_gdf,
    output_dir,
    tile_id,
    tile_no,
    coastal_closing_pixels,
):
    """Reconstruct and export the masks used by contours_preprocess.

    This mirrors the active Scotland settings:

        mask_landcover=False
        mask_ndwi=True
        mask_temporal=True
        buffer_pixels=33
        morphological closing=coastal_closing_pixels

    The production shoreline mask still comes from contours_preprocess.
    These products are QA diagnostics only.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "mask_diagnostics_summary.csv"

    expected = [
        output_dir / "01_all_time_land.tif",
        output_dir / "02_all_time_ocean_connected.tif",
        output_dir / "03_coastal_zone_class.tif",
        summary_file,
    ]

    if all(path.exists() for path in expected):
        print("Mask diagnostics already exist; skipping")
        return

    print("Building mask diagnostics...")

    # Mirror contours_preprocess: use gapfill where annual count <= 5,
    # then remove pixels with <= 1 observations.
    diagnostic_ds = yearly_ds.where(
        yearly_ds["count"] > 5,
        gapfill_ds,
    )
    diagnostic_ds = diagnostic_ds.where(
        diagnostic_ds["count"] > 1
    )

    # DEA convention at this stage: True = land, False = water.
    thresholded_ds = (
        diagnostic_ds[WATER_INDEX] < INDEX_THRESHOLD
    )
    nodata = diagnostic_ds[WATER_INDEX].isnull()
    thresholded_ds = thresholded_ds.where(~nodata)

    # Mirror mask_ndwi=True.
    ndwi_land = diagnostic_ds["ndwi"] < 0
    ndwi_mask = odc.algo.mask_cleanup(
        ndwi_land,
        mask_filters=[("dilation", 2)],
    )
    ndwi_mask = ndwi_mask.where(~nodata, 1)
    thresholded_ds = thresholded_ds.where(
        ndwi_mask,
        0,
    )

    # Mirror mask_temporal=True.
    temporal_mask = temporal_masking(
        thresholded_ds == 1
    )
    thresholded_ds = thresholded_ds.where(
        temporal_mask,
        0,
    )

    # Persistent land definition used by DEA Coastlines.
    all_time_land = (
        thresholded_ds.mean(dim="year") >= 0.15
    )

    # Show the connected ocean after the active Scotland closing distance.
    closed_land = xr.apply_ufunc(
        lambda arr: binary_closing(
            arr,
            footprint=disk(coastal_closing_pixels),
        ),
        all_time_land,
    )

    all_time_ocean = ocean_masking(
        closed_land,
        tide_points_gdf,
    )

    # Exact categorical coastal mask used internally by DEA:
    #   0 = connected ocean outside coastal zone
    #   1 = coastal analysis zone
    #   2 = inland / disconnected water and land interior
    coastal_class = coastal_masking(
        ds=all_time_land,
        tide_points_gdf=tide_points_gdf,
        buffer=33,
        closing=coastal_closing_pixels,
    )

    inland_mask = coastal_class == 2
    coastal_zone = coastal_class == 1

    # Mirror DEA's annual ocean-connectivity step.
    annual_ocean = (
        (thresholded_ds != 0)
        .where(~inland_mask, 1)
        .groupby("year")
        .apply(
            lambda x: ocean_masking(
                x,
                tide_points_gdf,
                1,
                3,
            )
        )
    )

    years = annual_ocean["year"].values.astype(int)
    diagnostic_year = (
        DIAGNOSTIC_YEAR
        if DIAGNOSTIC_YEAR in years
        else int(
            years[
                np.argmin(
                    np.abs(years - DIAGNOSTIC_YEAR)
                )
            ]
        )
    )

    if diagnostic_year != DIAGNOSTIC_YEAR:
        print(
            f"Requested diagnostic year {DIAGNOSTIC_YEAR} unavailable; "
            f"using {diagnostic_year}"
        )

    annual_ocean_year = annual_ocean.sel(
        year=diagnostic_year
    )
    ndwi_mask_year = ndwi_mask.sel(
        year=diagnostic_year
    )
    temporal_mask_year = temporal_mask.sel(
        year=diagnostic_year
    )

    # Final spatial validity mask controlling where contours may be extracted
    # for the representative year.
    final_valid_mask = (
        annual_ocean_year
        & coastal_zone
        & ndwi_mask_year
        & temporal_mask_year
    )

    reference = diagnostic_ds[WATER_INDEX].isel(
        year=0
    )

    write_mask_geotiff(
        all_time_land,
        output_dir / "01_all_time_land.tif",
        reference,
    )
    write_mask_geotiff(
        all_time_ocean,
        output_dir / "02_all_time_ocean_connected.tif",
        reference,
    )
    write_mask_geotiff(
        coastal_class,
        output_dir / "03_coastal_zone_class.tif",
        reference,
    )
    write_mask_geotiff(
        annual_ocean_year,
        output_dir
        / f"04_annual_ocean_connected_{diagnostic_year}.tif",
        reference,
    )
    write_mask_geotiff(
        final_valid_mask,
        output_dir
        / f"05_final_valid_mask_{diagnostic_year}.tif",
        reference,
    )

    total_pixels = int(
        all_time_land.sizes["y"]
        * all_time_land.sizes["x"]
    )

    summary = pd.DataFrame(
        [
            {
                "tile_no": tile_no,
                "tile_id": tile_id,
                "diagnostic_year": diagnostic_year,
                "n_ocean_seed_points": len(tide_points_gdf),
                "all_time_land_fraction": float(
                    all_time_land.mean().values
                ),
                "all_time_ocean_fraction": float(
                    all_time_ocean.mean().values
                ),
                "coastal_zone_fraction": float(
                    coastal_zone.mean().values
                ),
                "inland_fraction": float(
                    inland_mask.mean().values
                ),
                "annual_ocean_fraction": float(
                    annual_ocean_year.mean().values
                ),
                "final_valid_fraction": float(
                    final_valid_mask.mean().values
                ),
                "n_pixels": total_pixels,
                "closing_pixels": coastal_closing_pixels,
                "closing_metres_nominal": coastal_closing_pixels * RESOLUTION,
                "coastal_buffer_pixels": 33,
                "coastal_buffer_metres_nominal": 33 * RESOLUTION,
            }
        ]
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print(f"Mask diagnostics written to: {output_dir}")



# =============================================================================
# DEA CHANGE ANALYSIS + MERGED PRODUCTS
# =============================================================================


def rate_output_paths(tile_id):
    """Return per-tile DEA change-analysis output paths."""
    _, _, vector_dir = versioned_paths(tile_id)

    stem = (
        f"ratesofchange_{tile_id}_{VECTOR_VERSION}_"
        f"{WATER_INDEX}_{INDEX_THRESHOLD:.2f}_baseline{BASELINE_YEAR}"
    )

    return (
        vector_dir / f"{stem}_BNG.gpkg",
        vector_dir / f"{stem}.geojson",
    )


def analyse_tile_rates(
    contours,
    yearly_ds,
    core_geometry,
    tile_id,
    tile_no,
    vector_dir,
):
    """Run DEA's native baseline-point analysis for one buffered tile.

    The annual shoreline contours and water-index rasters are the buffered
    processing products, so points near a 20 km core boundary retain the same
    surrounding context used during shoreline extraction. Analysis points are
    clipped back to the authoritative core only after all annual distances have
    been calculated.

    The calculation itself uses DEA Coastlines' native:
        points_on_line()
        annual_movements()
        calculate_regressions()
        all_time_stats()
    """
    output_gpkg, output_geojson = rate_output_paths(tile_id)

    if output_gpkg.exists() and output_geojson.exists():
        print(f"Rate analysis already complete: {output_geojson}")
        return gpd.read_file(output_gpkg).to_crs(CRS)

    analysis_contours = contours.copy()
    analysis_contours.index = analysis_contours.index.astype(str)
    analysis_contours.index.name = "year"

    baseline_key = str(BASELINE_YEAR)

    if baseline_key not in analysis_contours.index:
        print(
            f"No {BASELINE_YEAR} baseline shoreline for {tile_id}; "
            "skipping rate analysis."
        )
        return None

    print(
        f"Running DEA change analysis using {BASELINE_YEAR} baseline "
        f"and {POINT_SPACING_M} m points..."
    )

    try:
        points = points_on_line(
            analysis_contours,
            baseline_key,
            distance=POINT_SPACING_M,
        )
    except KeyError:
        print(
            f"Unable to generate baseline points for {tile_id}; "
            "skipping rate analysis."
        )
        return None

    if points is None or points.empty:
        print(f"No baseline points generated for {tile_id}; skipping.")
        return None

    points = annual_movements(
        points_gdf=points,
        contours_gdf=analysis_contours,
        yearly_ds=yearly_ds,
        baseline_year=baseline_key,
        water_index=WATER_INDEX,
        max_valid_dist=MAX_VALID_DIST_M,
    )

    points = calculate_regressions(
        points,
        analysis_contours,
    )

    stats_fields = [
        "valid_obs",
        "valid_span",
        "sce",
        "nsm",
        "max_year",
        "min_year",
    ]

    points[stats_fields] = points.apply(
        lambda row: all_time_stats(
            row,
            initial_year=START_YEAR,
        ),
        axis=1,
    )

    # Retain DEA's portable quality checks. The Africa-specific offshore-island
    # and geomorphology flags are deliberately not applied here.
    points["certainty"] = "good"
    points.loc[
        points["rate_time"].abs() > 200,
        "certainty",
    ] = "extreme value (> 200 m)"
    points.loc[
        points["angle_std"] > 30,
        "certainty",
    ] = "high angular variability"
    points.loc[
        points["valid_obs"] < 15,
        "certainty",
    ] = "insufficient observations"

    # Authoritative ownership: only points inside this 20 km core are exported.
    points = points[
        points.geometry.intersects(core_geometry)
    ].copy()

    if points.empty:
        print(
            f"No {BASELINE_YEAR} analysis points remain inside "
            f"core tile {tile_id}."
        )
        return None

    points = points.reset_index(drop=True)

    points["tile_no"] = tile_no
    points["tile_id"] = tile_id
    points["indicator"] = "DEA_MSL"
    points["baseline_year"] = BASELINE_YEAR
    points["point_spacing_m"] = POINT_SPACING_M
    points["water_index"] = WATER_INDEX
    points["threshold"] = INDEX_THRESHOLD
    points["tide_model"] = TIDE_MODEL
    points["tide_datum"] = "model MSL"
    points["processing_buffer_m"] = PROCESSING_BUFFER_M
    points["coastal_closing_pixels"] = COASTAL_CLOSING_OVERRIDES.get(
        tile_no,
        COASTAL_CLOSING_DEFAULT,
    )
    points["raster_version"] = RASTER_VERSION
    points["vector_version"] = VECTOR_VERSION

    points["point_no"] = np.arange(1, len(points) + 1)

    points["point_id"] = (
        points["tile_id"].astype(str)
        + "_P"
        + points["point_no"].astype(str).str.zfill(5)
    )

    # Put identifiers and headline statistics first while retaining all
    # dist_YYYY fields required by CoastalMappingViewer timeseries popups.
    preferred = [
        "point_id",
        "tile_no",
        "tile_id",
        "point_no",
        "indicator",
        "baseline_year",
        "point_spacing_m",
        "rate_time",
        "sig_time",
        "se_time",
        "valid_obs",
        "valid_span",
        "sce",
        "nsm",
        "max_year",
        "min_year",
        "angle_mean",
        "angle_std",
        "certainty",
        "water_index",
        "threshold",
        "tide_model",
        "tide_datum",
        "processing_buffer_m",
        "coastal_closing_pixels",
        "raster_version",
        "vector_version",
    ]

    dist_fields = sorted(
        [
            column
            for column in points.columns
            if column.startswith("dist_")
        ],
        key=lambda name: int(name.split("_", 1)[1]),
    )

    ordered = (
        [column for column in preferred if column in points.columns]
        + dist_fields
        + ["geometry"]
    )

    points = points[ordered]

    vector_dir.mkdir(parents=True, exist_ok=True)

    points.to_file(
        output_gpkg,
        layer="rates_of_change",
        driver="GPKG",
    )

    points.to_crs(WGS84).to_file(
        output_geojson,
        driver="GeoJSON",
    )

    print(f"  Rate points: {len(points):,}")
    print(f"  BNG GPKG:    {output_gpkg}")
    print(f"  Web GeoJSON: {output_geojson}")

    return points


def analyse_existing_tile(row):
    """Add DEA rate outputs to an already-extracted tile without redoing STAC/tides.

    This path is used when annual shoreline vectors already exist but the
    rates-of-change outputs do not. It reloads the completed annual/gapfill
    rasters, reconstructs the buffered shoreline contours, and runs only the
    vector analysis stages.
    """
    tile_id = str(row[TILE_ID_FIELD])
    tile_no = int(row[TILE_NO_FIELD])
    core_geometry = row.geometry

    coastal_closing_pixels = COASTAL_CLOSING_OVERRIDES.get(
        tile_no,
        COASTAL_CLOSING_DEFAULT,
    )

    _, _, vector_dir = versioned_paths(tile_id)

    print(
        f"Shorelines already exist for {tile_id}; "
        "running analysis from existing rasters."
    )

    yearly_ds, gapfill_ds = load_rasters(
        path=str(RASTER_ROOT),
        raster_version=RASTER_VERSION,
        study_area=tile_id,
        water_index=WATER_INDEX,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    seed_file = vector_dir / "ocean_seed_points.gpkg"

    if seed_file.exists():
        seeds = gpd.read_file(seed_file).to_crs(yearly_ds.rio.crs)
    else:
        seeds = ocean_seeds(yearly_ds)
        seeds.to_file(
            seed_file,
            driver="GPKG",
        )

    masked_ds, _ = contours_preprocess(
        yearly_ds=yearly_ds,
        gapfill_ds=gapfill_ds,
        water_index=WATER_INDEX,
        index_threshold=INDEX_THRESHOLD,
        tide_points_gdf=seeds,
        buffer_pixels=33,
        coastal_closing_pixels=coastal_closing_pixels,
        mask_landcover=False,
        mask_ndwi=True,
        mask_temporal=True,
        mask_modifications=None,
    )

    contours = subpixel_contours(
        da=masked_ds,
        z_values=INDEX_THRESHOLD,
        min_vertices=10,
        dim="year",
    ).set_index("year")

    return analyse_tile_rates(
        contours=contours,
        yearly_ds=yearly_ds,
        core_geometry=core_geometry,
        tile_id=tile_id,
        tile_no=tile_no,
        vector_dir=vector_dir,
    )


def selected_output_label(tiles, national=False):
    """Return a stable label for pilot, national or ad-hoc selections."""
    if national:
        return "Scotland"

    tile_numbers = tuple(
        sorted(tiles[TILE_NO_FIELD].astype(int).tolist())
    )

    pilot_numbers = tuple(sorted(PILOT_TILE_NOS or []))

    if tile_numbers == pilot_numbers:
        return "Scotland_pilot11"

    return "Scotland_tiles_" + "_".join(
        str(value)
        for value in tile_numbers
    )


def merge_completed_shorelines(tiles):
    """Merge completed core-clipped annual shoreline GPKGs for selected tiles."""
    frames = []
    missing = []

    for _, row in tiles.iterrows():
        tile_id = str(row[TILE_ID_FIELD])
        _, _, vector_dir = versioned_paths(tile_id)

        shoreline_path = (
            vector_dir
            / (
                f"annualshorelines_{tile_id}_{VECTOR_VERSION}_"
                f"{WATER_INDEX}_{INDEX_THRESHOLD:.2f}_BNG.gpkg"
            )
        )

        if not shoreline_path.exists():
            missing.append(tile_id)
            continue

        frame = gpd.read_file(shoreline_path).to_crs(CRS)

        if frame.empty:
            continue

        frames.append(frame)

    if not frames:
        raise RuntimeError(
            "No completed shoreline GPKGs found for the selected tiles."
        )

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=CRS,
    )

    merged = merged[
        merged.geometry.notnull()
        & ~merged.geometry.is_empty
    ].copy()

    if "shoreline_id" in merged.columns:
        merged = merged.drop_duplicates(
            subset="shoreline_id",
            keep="first",
        )

    if missing:
        print(
            "No shoreline output for selected tile(s): "
            + ", ".join(missing)
        )

    return merged


def merge_completed_rates(tiles):
    """Merge completed per-tile DEA rate-point GPKGs for selected tiles."""
    frames = []
    missing = []

    for _, row in tiles.iterrows():
        tile_id = str(row[TILE_ID_FIELD])
        output_gpkg, _ = rate_output_paths(tile_id)

        if not output_gpkg.exists():
            missing.append(tile_id)
            continue

        frame = gpd.read_file(output_gpkg).to_crs(CRS)

        if frame.empty:
            continue

        frames.append(frame)

    if not frames:
        print("No completed rate-point GPKGs found for selected tiles.")
        return None

    merged = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        crs=CRS,
    )

    merged = merged[
        merged.geometry.notnull()
        & ~merged.geometry.is_empty
    ].copy()

    if "point_id" in merged.columns:
        merged = merged.drop_duplicates(
            subset="point_id",
            keep="first",
        )

    if missing:
        print(
            "No rates output for selected tile(s): "
            + ", ".join(missing)
        )

    return merged


def write_merged_products(
    shorelines,
    rates,
    tiles,
    national=False,
):
    """Write merged scientific, web and CMT-handoff products.

    GPKG outputs remain in EPSG:27700 and are the analysis/master products.
    GeoJSON outputs are transformed to EPSG:4326 for CoastalMappingViewer.
    No dissolve is applied across tile boundaries, preserving source provenance.
    """
    label = selected_output_label(
        tiles,
        national=national,
    )

    output_dir = (
        ANALYSIS_ROOT
        / VECTOR_VERSION
        / label
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    shoreline_stem = (
        f"{label}_DEA_MSL_annual_shorelines_"
        f"{START_YEAR}_{END_YEAR}"
    )

    shoreline_gpkg = output_dir / f"{shoreline_stem}_BNG.gpkg"
    shoreline_geojson = output_dir / f"{shoreline_stem}.geojson"

    shorelines.to_file(
        shoreline_gpkg,
        layer="annual_shorelines",
        driver="GPKG",
    )

    shorelines.to_crs(WGS84).to_file(
        shoreline_geojson,
        driver="GeoJSON",
    )

    # Explicit CMT handoff: the latest mature MSL shoreline is retained as a
    # baseline candidate. It is not smoothed or converted to transects here.
    baseline = shorelines[
        shorelines["year"].astype(int) == BASELINE_YEAR
    ].copy()

    baseline_gpkg = (
        output_dir
        / f"{label}_CMT_baseline_candidate_DEA_MSL_{BASELINE_YEAR}_BNG.gpkg"
    )

    baseline_geojson = (
        output_dir
        / f"{label}_CMT_baseline_candidate_DEA_MSL_{BASELINE_YEAR}.geojson"
    )

    if not baseline.empty:
        baseline.to_file(
            baseline_gpkg,
            layer="baseline_candidate",
            driver="GPKG",
        )

        baseline.to_crs(WGS84).to_file(
            baseline_geojson,
            driver="GeoJSON",
        )

    print()
    print("Merged annual shoreline products:")
    print(f"  BNG GPKG:    {shoreline_gpkg}")
    print(f"  Web GeoJSON: {shoreline_geojson}")

    if not baseline.empty:
        print(f"  CMT baseline candidate: {baseline_gpkg}")

    if rates is None or rates.empty:
        return

    rates_stem = (
        f"{label}_DEA_MSL_ratesofchange_"
        f"baseline{BASELINE_YEAR}"
    )

    rates_gpkg = output_dir / f"{rates_stem}_BNG.gpkg"
    rates_geojson = output_dir / f"{rates_stem}.geojson"

    rates.to_file(
        rates_gpkg,
        layer="rates_of_change",
        driver="GPKG",
    )

    rates.to_crs(WGS84).to_file(
        rates_geojson,
        driver="GeoJSON",
    )

    print()
    print("Merged DEA change-analysis products:")
    print(f"  BNG GPKG:    {rates_gpkg}")
    print(f"  Web GeoJSON: {rates_geojson}")
    print(f"  Rate points: {len(rates):,}")


def run_analysis(tiles, national=False):
    """Merge selected tile products after per-tile DEA analysis has completed."""
    print()
    print("Merging completed annual shorelines...")
    shorelines = merge_completed_shorelines(
        tiles
    )

    print(
        f"  {len(shorelines):,} shoreline features "
        f"across {shorelines['year'].nunique()} years"
    )

    print("Merging completed DEA rate points...")
    rates = merge_completed_rates(
        tiles
    )

    write_merged_products(
        shorelines=shorelines,
        rates=rates,
        tiles=tiles,
        national=national,
    )

# =============================================================================
# ONE GRID TILE
# =============================================================================


def run_tile(row, fresh=False, analyse=False):
    """Run the full buffered raster -> core-clipped vector workflow.

    If analyse=True, DEA's native baseline-point change analysis is also
    generated for this tile.
    """

    tile_id = str(row[TILE_ID_FIELD])
    tile_no = int(row[TILE_NO_FIELD])

    # Use the DEA default closing distance unless a tile-specific
    # override has been defined.
    coastal_closing_pixels = COASTAL_CLOSING_OVERRIDES.get(
        tile_no,
        COASTAL_CLOSING_DEFAULT,
    )

    core_geometry = row.geometry
    process_geometry = processing_footprint(core_geometry)
    bbox = wgs84_bbox(process_geometry)

    print()
    print("=" * 78)
    print(f"Tile no:         {tile_no}")
    print(f"Tile ID:         {tile_id}")
    print(f"Core:            20 x 20 km")
    print(f"Buffer:          {PROCESSING_BUFFER_M / 1000:.1f} km")
    print(f"Processing area: 30 x 30 km")
    print(f"Raster version:  {RASTER_VERSION}")
    print(f"Vector version:  {VECTOR_VERSION}")
    print("=" * 78)

    if fresh:
        clear_current_version(tile_id)

    cache, raster_dir, vector_dir = versioned_paths(tile_id)

    dem_file = cache / "cop_dem_glo30_bng.tif"
    cutoff_file = (
        cache
        / f"{TIDE_MODEL}_cutoffs_{CUTOFF_START}_{CUTOFF_END}.nc"
    )

    tide_dir = cache / "tides"
    obs_dir = cache / "tidal_observations"

    for folder in [
        cache,
        tide_dir,
        obs_dir,
        raster_dir,
        vector_dir,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    output_geojson = (
        vector_dir
        / (
            f"annualshorelines_{tile_id}_{VECTOR_VERSION}_"
            f"{WATER_INDEX}_{INDEX_THRESHOLD:.2f}.geojson"
        )
    )

    output_gpkg = (
        vector_dir
        / (
            f"annualshorelines_{tile_id}_{VECTOR_VERSION}_"
            f"{WATER_INDEX}_{INDEX_THRESHOLD:.2f}_BNG.gpkg"
        )
    )

    rate_gpkg, rate_geojson = rate_output_paths(tile_id)

    if output_geojson.exists() and not fresh:
        print(f"Already complete: {output_geojson}")

        if analyse and not (
            rate_gpkg.exists()
            and rate_geojson.exists()
        ):
            analyse_existing_tile(row)

        return {
            "tile_no": tile_no,
            "tile_id": tile_id,
            "status": "already_complete",
            "error": "",
        }

    # -------------------------------------------------------------------------
    # STAGE 1: buffered DEM + Landsat loader
    # -------------------------------------------------------------------------

    if dem_file.exists():
        print("Using cached Copernicus GLO-30 DEM")
        dem = (
            rioxarray.open_rasterio(dem_file)
            .squeeze("band", drop=True)
            .load()
        )

    else:
        print("Loading Copernicus GLO-30 DEM over buffered tile...")
        dem = load_dem_stac(
            bbox,
            output_crs=CRS,
            resolution=RESOLUTION,
        ).compute()
        dem.rio.to_raster(dem_file)

    def load_year(year, shadow=False, fail_on_error=True):
        return load_water_index_stac(
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            output_crs=CRS,
            resolution=RESOLUTION,
            cloud_cover=CLOUD_COVER,
            platforms=PLATFORMS,
            fail_on_error=fail_on_error,
            dem=dem if shadow else None,
            mask_terrain_shadow=shadow,
            terrain_shadow_threshold=SHADOW_THRESHOLD,
            terrain_shadow_radius=SHADOW_RADIUS,
        )

    # -------------------------------------------------------------------------
    # STAGE 2: tile-specific tidal cutoffs over buffered footprint
    # -------------------------------------------------------------------------

    if cutoff_file.exists():
        print("Using cached tile-specific tidal cutoffs")
        with xr.open_dataset(cutoff_file) as ds:
            cutoff_min = ds["tide_cutoff_min"].load()
            cutoff_max = ds["tide_cutoff_max"].load()

    else:
        print("Building tile-specific tidal cutoffs...")

        record = load_water_index_stac(
            bbox=bbox,
            datetime=f"{CUTOFF_START}-01-01/{CUTOFF_END}-12-31",
            output_crs=CRS,
            resolution=RESOLUTION,
            cloud_cover=CLOUD_COVER,
            platforms=PLATFORMS,
        )

        tides_lowres = pixel_tides(
            data=record,
            model=TIDE_MODEL,
            directory=TIDE_MODEL_DIR,
            resample=False,
        ).compute()

        cutoff_min, cutoff_max = tide_cutoffs(
            ds=load_year(START_YEAR),
            tides_lowres=tides_lowres,
            tide_centre=0.0,
        )

        cutoff_ds = xr.Dataset(
            {
                "tide_cutoff_min": cutoff_min,
                "tide_cutoff_max": cutoff_max,
            },
            attrs={
                "tile_id": tile_id,
                "tile_no": tile_no,
                "tide_model": TIDE_MODEL,
                "cutoff_start_year": CUTOFF_START,
                "cutoff_end_year": CUTOFF_END,
                "processing_buffer_m": PROCESSING_BUFFER_M,
                "processing_crs": CRS,
            },
        )

        save_netcdf_safe(cutoff_ds, cutoff_file)

    # -------------------------------------------------------------------------
    # STAGE 3: yearly buffered, shadow-masked, tide-filtered observations
    # -------------------------------------------------------------------------

    def observations(year):
        obs_file = obs_dir / f"{year}_tidal.nc"
        tide_file = tide_dir / f"{year}_tides.nc"

        if obs_file.exists():
            with xr.open_dataset(obs_file) as ds:
                return ds.load()

        if tide_file.exists():
            with xr.open_dataset(tide_file) as ds:
                tides = ds["tide_m"].load()

        else:
            print(f"{year}: modelling {TIDE_MODEL} tides")
            tides = pixel_tides(
                data=load_year(year),
                model=TIDE_MODEL,
                directory=TIDE_MODEL_DIR,
                resample=True,
            ).rename("tide_m").compute()

            save_netcdf_safe(tides, tide_file)

        for attempt in range(1, 4):
            pc_sas.TOKEN_CACHE.clear()

            try:
                print(
                    f"{year}: loading buffered Landsat "
                    f"(attempt {attempt}/3)"
                )

                ds = load_year(
                    year,
                    shadow=True,
                    fail_on_error=(attempt < 3),
                )

                image_times = set(
                    ds.time.values.astype("datetime64[ns]")
                )
                tide_times = set(
                    tides.time.values.astype("datetime64[ns]")
                )

                if not image_times.issubset(tide_times):
                    print(
                        f"{year}: Landsat timestamps changed; "
                        "rebuilding tide cache"
                    )

                    tides = pixel_tides(
                        data=ds,
                        model=TIDE_MODEL,
                        directory=TIDE_MODEL_DIR,
                        resample=True,
                    ).rename("tide_m").compute()

                    save_netcdf_safe(tides, tide_file)

                ds = ds[["mndwi", "ndwi"]]
                ds["tide_m"] = tides.sel(time=ds.time)

                like = ds.tide_m.isel(time=0)

                filtered = load_tidal_subset(
                    ds,
                    cutoff_min.interp_like(like),
                    cutoff_max.interp_like(like),
                )

                save_netcdf_safe(filtered, obs_file)

                return filtered

            except Exception:
                if attempt == 3:
                    raise

                time.sleep(5)

    # -------------------------------------------------------------------------
    # STAGE 4: annual + true three-year gapfill rasters
    # -------------------------------------------------------------------------

    print("Creating/reusing annual raster composites...")

    previous = observations(START_YEAR - 1)
    current = observations(START_YEAR)
    future = observations(START_YEAR + 1)

    for year in range(START_YEAR, END_YEAR + 1):

        if not rasters_exist(raster_dir, year):
            tidal_composite(
                current,
                year,
                "year",
                str(raster_dir),
                export_geotiff=True,
            )

        if not rasters_exist(
            raster_dir,
            year,
            "_gapfill",
        ):
            gapfill = xr.concat(
                [previous, current, future],
                dim="time",
            ).sortby("time")

            tidal_composite(
                gapfill,
                year,
                "year",
                str(raster_dir),
                output_suffix="_gapfill",
                export_geotiff=True,
            )

        if year < END_YEAR:
            previous = current
            current = future
            future = observations(year + 2)

    # -------------------------------------------------------------------------
    # STAGE 5: shoreline extraction over buffered processing footprint
    # -------------------------------------------------------------------------

    print("Extracting buffered annual MSL shorelines...")

    yearly_ds, gapfill_ds = load_rasters(
        path=str(RASTER_ROOT),
        raster_version=RASTER_VERSION,
        study_area=tile_id,
        water_index=WATER_INDEX,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    seeds = ocean_seeds(yearly_ds)

    seeds.to_file(
        vector_dir / "ocean_seed_points.gpkg",
        driver="GPKG",
    )

    if EXPORT_MASK_DIAGNOSTICS:
        export_mask_diagnostics(
            yearly_ds=yearly_ds,
            gapfill_ds=gapfill_ds,
            tide_points_gdf=seeds,
            output_dir=vector_dir / "mask_diagnostics",
            tile_id=tile_id,
            tile_no=tile_no,
            coastal_closing_pixels=coastal_closing_pixels,
        )

    masked_ds, certainty = contours_preprocess(
        yearly_ds=yearly_ds,
        gapfill_ds=gapfill_ds,
        water_index=WATER_INDEX,
        index_threshold=INDEX_THRESHOLD,
        tide_points_gdf=seeds,
        buffer_pixels=33,
        coastal_closing_pixels=coastal_closing_pixels,
        mask_landcover=False,
        mask_ndwi=True,
        mask_temporal=True,
        mask_modifications=None,
    )

    contours = subpixel_contours(
        da=masked_ds,
        z_values=INDEX_THRESHOLD,
        min_vertices=10,
        dim="year",
    ).set_index("year")

    # DEA change analysis must use the unsplit buffered contours. Certainty
    # attribution below can split a year's shoreline into multiple geometries,
    # whereas annual_movements expects one annual geometry.
    if analyse:
        analyse_tile_rates(
            contours=contours,
            yearly_ds=yearly_ds,
            core_geometry=core_geometry,
            tile_id=tile_id,
            tile_no=tile_no,
            vector_dir=vector_dir,
        )

    contours.index = contours.index.astype(str)

    contours = contour_certainty(
        contours,
        {
            str(year): mask
            for year, mask in certainty.items()
        },
    )

    contours.index = contours.index.astype(int)
    contours.index.name = "year"

    shorelines = contours.reset_index()

    shorelines = shorelines[
        shorelines.geometry.notnull()
        & ~shorelines.geometry.is_empty
    ].copy()

    shorelines = shorelines.to_crs(CRS)

    # -------------------------------------------------------------------------
    # STAGE 6: ownership clip back to authoritative 20 km CORE
    # -------------------------------------------------------------------------

    print("Clipping shorelines back to 20 km core tile...")

    shorelines["geometry"] = (
        shorelines.geometry.intersection(core_geometry)
    )

    shorelines = shorelines[
        shorelines.geometry.notnull()
        & ~shorelines.geometry.is_empty
    ].copy()

    if shorelines.empty:
        raise RuntimeError(
            "No shoreline geometry remains after clipping to the core tile"
        )

    # Tile and provenance attributes.
    shorelines["tile_no"] = tile_no
    shorelines["tile_id"] = tile_id
    shorelines["indicator"] = "DEA_MSL"
    shorelines["water_index"] = WATER_INDEX
    shorelines["threshold"] = INDEX_THRESHOLD
    shorelines["tide_model"] = TIDE_MODEL
    shorelines["tide_datum"] = "model MSL"
    shorelines["processing_buffer_m"] = PROCESSING_BUFFER_M
    shorelines["coastal_closing_pixels"] = coastal_closing_pixels
    shorelines["raster_version"] = RASTER_VERSION
    shorelines["vector_version"] = VECTOR_VERSION

    shorelines["segment_no"] = (
        shorelines.groupby("year").cumcount() + 1
    )

    # Nationally unique source-feature ID.
    shorelines["shoreline_id"] = (
        shorelines["tile_id"].astype(str)
        + "_"
        + shorelines["year"].astype(str)
        + "_"
        + shorelines["segment_no"].astype(str).str.zfill(3)
    )

    # BNG source product.
    shorelines.to_file(
        output_gpkg,
        driver="GPKG",
    )

    # WGS84 web/handoff product.
    shorelines.to_crs(WGS84).to_file(
        output_geojson,
        driver="GeoJSON",
    )

    cache_gib = directory_size_gib(cache)
    raster_gib = directory_size_gib(raster_dir)
    vector_gib = directory_size_gib(vector_dir)

    print(f"Finished tile {tile_no}: {tile_id}")
    print(f"  Cache:   {cache_gib:.2f} GiB")
    print(f"  Rasters: {raster_gib:.2f} GiB")
    print(f"  Vectors: {vector_gib:.3f} GiB")
    print(f"  Output:  {output_geojson}")

    return {
        "tile_no": tile_no,
        "tile_id": tile_id,
        "status": "completed",
        "error": "",
        "cache_gib": round(cache_gib, 3),
        "raster_gib": round(raster_gib, 3),
        "vector_gib": round(vector_gib, 3),
    }



# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    if not TIDE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Tide model directory not found: {TIDE_MODEL_DIR}"
        )

    tiles = validate_and_select_grid(args)

    print()
    print("=" * 78)
    print("SCOTLAND DEA COASTLINES — 20 KM GRID DRIVER")
    print("=" * 78)
    print(f"Grid file:       {GRID_FILE}")
    print(f"Tiles selected:  {len(tiles)}")
    print(
        "Tile numbers:    "
        + ", ".join(
            str(value)
            for value in tiles[TILE_NO_FIELD].astype(int)
        )
    )
    print(f"Core size:       20 x 20 km")
    print(f"Buffer:          {PROCESSING_BUFFER_M / 1000:.1f} km")
    print(f"Working extent:  30 x 30 km")
    print(f"Raster version:  {RASTER_VERSION}")
    print(f"Vector version:  {VECTOR_VERSION}")
    print(f"Fresh run:       {args.fresh}")
    print(f"Run analysis:    {args.analyse}")
    print(f"Years:           {START_YEAR}-{END_YEAR}")
    print(f"Tide model:      {TIDE_MODEL}")
    print("=" * 78)

    results = []

    for _, row in tiles.iterrows():
        tile_no = int(row[TILE_NO_FIELD])
        tile_id = str(row[TILE_ID_FIELD])

        try:
            result = run_tile(
                row,
                fresh=args.fresh,
                analyse=args.analyse,
            )
            results.append(result)

        except Exception as error:
            print(
                f"ERROR: tile {tile_no} ({tile_id}) failed: {error}"
            )

            results.append(
                {
                    "tile_no": tile_no,
                    "tile_id": tile_id,
                    "status": "failed",
                    "error": str(error),
                }
            )

    # Write one independent status CSV per tile.
    # This is safe when multiple tile-processing scripts run in parallel.
    status_dir = (
        LOG_ROOT
        / RASTER_VERSION
        / VECTOR_VERSION
        / "tiles"
    )

    status_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for result in results:

        tile_id = result["tile_id"]

        status_file = (
            status_dir
            / f"{tile_id}.csv"
        )

        pd.DataFrame(
            [result]
        ).to_csv(
            status_file,
            index=False,
        )

    completed = sum(
        result["status"] in {"completed", "already_complete"}
        for result in results
    )

    failed = sum(
        result["status"] == "failed"
        for result in results
    )

    print()
    print("=" * 78)
    print("GRID RUN COMPLETE")
    print("=" * 78)
    print(f"Successful/already complete: {completed}")
    print(f"Failed:                      {failed}")
    print(f"Per-tile status directory:   {status_dir}")

    if failed:
        print()
        print("Failed tiles:")
        for result in results:
            if result["status"] == "failed":
                print(
                    f"  {result['tile_no']} "
                    f"({result['tile_id']}): "
                    f"{result['error']}"
                )

    if args.analyse:

        print()
        print("=" * 78)
        print("RUNNING POST-PROCESSING AND CHANGE ANALYSIS")
        print("=" * 78)

        run_analysis(
            tiles=tiles,
            national=args.all,
        )

if __name__ == "__main__":
    main()
