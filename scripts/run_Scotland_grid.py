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
from skimage.morphology import binary_closing, disk

from coastlines.raster import load_tidal_subset, tidal_composite, tide_cutoffs
from coastlines.stac import load_dem_stac, load_water_index_stac
from coastlines.vector import (
    coastal_masking,
    contour_certainty,
    contours_preprocess,
    load_rasters,
    ocean_masking,
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

TIDE_MODEL = "EOT20"
PLATFORMS = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
CLOUD_COVER = 80

SHADOW_THRESHOLD = 0.5
SHADOW_RADIUS = 1

WATER_INDEX = "mndwi"
INDEX_THRESHOLD = 0.0

# Pilot QA: export diagnostic masks without changing the shoreline method.
EXPORT_MASK_DIAGNOSTICS = True
DIAGNOSTIC_YEAR = 2020


# =============================================================================
# VERSIONING
# =============================================================================

# Keep this pilot distinct from the earlier AOI versions. If the regional pilot
# leads to methodological changes, increment this before the national run.
RASTER_VERSION = "scotland_grid_v0.1_bng_shadow"
VECTOR_VERSION = "scotland_grid_v0.1_bng_shadow"


# =============================================================================
# OUTPUT ROOTS
# =============================================================================

# Change RESULTS_ROOT to a mounted large drive before national production if
# desired, e.g. Path("/path/to/large_drive/dea_scotland").
RASTER_ROOT = RESULTS_FOLDER / "dea_rasters"
VECTOR_ROOT = RESULTS_FOLDER / "dea_vectors"
CACHE_ROOT = RESULTS_FOLDER / "dea_cache"
LOG_ROOT = RESULTS_FOLDER / "run_logs"


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

    out = out.rio.write_crs(reference.rio.crs, inplace=False)
    out.rio.write_transform(reference.rio.transform(), inplace=True)

    out.rio.to_raster(
        path,
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
):
    """Reconstruct and export the masks used by contours_preprocess.

    This mirrors the active Scotland settings:

        mask_landcover=False
        mask_ndwi=True
        mask_temporal=True
        buffer_pixels=33
        morphological closing=15 pixels

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

    # Show the connected ocean after DEA's 15-pixel (~450 m) closing.
    closed_land = xr.apply_ufunc(
        lambda arr: binary_closing(
            arr,
            footprint=disk(15),
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
        closing=15,
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
                "closing_pixels": 15,
                "closing_metres_nominal": 15 * RESOLUTION,
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
# ONE GRID TILE
# =============================================================================


def run_tile(row, fresh=False):
    """Run the full buffered raster -> core-clipped vector workflow."""

    tile_id = str(row[TILE_ID_FIELD])
    tile_no = int(row[TILE_NO_FIELD])

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

    if output_geojson.exists() and not fresh:
        print(f"Already complete: {output_geojson}")
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
        )

    masked_ds, certainty = contours_preprocess(
        yearly_ds=yearly_ds,
        gapfill_ds=gapfill_ds,
        water_index=WATER_INDEX,
        index_threshold=INDEX_THRESHOLD,
        tide_points_gdf=seeds,
        buffer_pixels=33,
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


if __name__ == "__main__":
    main()
