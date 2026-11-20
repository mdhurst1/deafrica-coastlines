#!/usr/bin/env python
"""Generate annual DEA-style MSL shorelines for Scottish exemplar AOIs.

The workflow is explicitly versioned at both raster and vector stages:

AOI -> DEM -> site-specific tidal cutoffs -> filtered Landsat -> annual/gapfill
rasters -> DEA preprocessing -> sub-pixel MSL shorelines.

Raster products are stored under RASTER_VERSION. Vector products are stored
under VECTOR_VERSION, allowing multiple vector experiments (e.g. different
water-index thresholds) to reuse the same raster stack.

By default the script is resumable. Pass --fresh to delete products for the
selected AOIs for the CURRENT raster/vector versions and rerun them from scratch.
"""

from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import planetary_computer.sas as pc_sas
import rioxarray
import xarray as xr
from dea_tools.spatial import subpixel_contours
from eo_tides.eo import pixel_tides
from shapely.geometry import Point

from coastlines.raster import load_tidal_subset, tidal_composite, tide_cutoffs
from coastlines.stac import load_dem_stac, load_sentinel2_indices_stac, load_water_index_stac
from coastlines.vector import contour_certainty, contours_preprocess, load_rasters

# =============================================================================
# USER SETTINGS
# =============================================================================

HOME_FOLDER = Path("/media/nevis_array/user_homes/mhurst/")
RESULTS_FOLDER = HOME_FOLDER / "scotland_dea_results"
AOI_FILE = HOME_FOLDER / "UKRI_Policy_Fellowship_Planet_AOIs_Exemplar_Sites.geojson"
TIDE_MODEL_DIR = HOME_FOLDER / "tide_models"

NAME_FIELD = "Name"

# None = run every AOI. For testing, e.g. ["St Fergus"] or ["Montrose Bay"]
SITES = None


# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

CRS = "EPSG:27700"  # British National Grid
RESOLUTION = 30

START_YEAR, END_YEAR = 1988, 2025
CUTOFF_START, CUTOFF_END = 1984, 2025
SENTINEL2_YEARS = (2019, 2024)

TIDE_MODEL = "FES2022_extrapolated"  # see eo-tides for other options
PLATFORMS = ["landsat-5", "landsat-7", "landsat-8", "landsat-9"]
CLOUD_COVER = 80

SHADOW_THRESHOLD = 0.5
SHADOW_RADIUS = 1

WATER_INDEX = "mndwi"
INDEX_THRESHOLD = 0.0

# -----------------------------------------------------------------------------
# VERSIONING
# -----------------------------------------------------------------------------
# RASTER_VERSION identifies everything that can alter the raster stack:
# CRS/resolution, satellite preprocessing, tide model/filtering, terrain shadow,
# annual compositing, etc.
#
# VECTOR_VERSION identifies the shoreline extraction experiment. It can be the
# same as RASTER_VERSION for the main production run, but can later be changed
# independently if testing thresholds/masking against the same rasters.

RASTER_VERSION = "v0.1.0"
VECTOR_VERSION = "v0.1.0"


# =============================================================================
# OUTPUT ROOTS
# =============================================================================

RASTER_ROOT = RESULTS_FOLDER / "outputs" / "dea_rasters"
VECTOR_ROOT = RESULTS_FOLDER / "outputs" / "dea_vectors"
CACHE_ROOT = RESULTS_FOLDER / "outputs" / "dea_cache"


# =============================================================================
# HELPERS
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run versioned DEA-style shoreline extraction for Scottish AOIs."
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete cached, raster and vector products for the selected AOIs "
            "for the CURRENT RASTER_VERSION/VECTOR_VERSION before running."
        ),
    )
    parser.add_argument("--sensor", choices=("landsat", "sentinel2"), default="landsat")
    parser.add_argument("--sites", nargs="+", metavar="SITE", help="Override the configured AOI names.")
    parser.add_argument("--tide-model", default=TIDE_MODEL,
                        help="Tide model name, for example EOT20 to match older Montrose outputs.")
    parser.add_argument("--years", nargs=2, type=int, metavar=("FIRST", "LAST"),
                        help="Override the Sentinel-2 annual output range (default 2019–2024).")
    args = parser.parse_args()
    if args.years is not None:
        if args.sensor != "sentinel2":
            parser.error("--years is only used with --sensor sentinel2")
        if not 2016 <= args.years[0] <= args.years[1]:
            parser.error("Sentinel-2 years must be ordered and start at 2016 or later for gapfill")
    return args


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")




def save_netcdf_safe(obj, path):
    """Write an xarray object to NetCDF after removing object-dtype metadata.

    ODC/STAC and eo-tides can attach auxiliary coordinates containing Python
    objects (for example CRS/model metadata). NetCDF4 cannot serialise these
    directly. They are not required by the cached Coastlines products, so
    non-dimensional object coordinates are dropped before writing. String-like
    dimension coordinates are converted to normal strings. The file is written
    atomically so a failed write cannot leave a corrupt cache behind.
    """
    if isinstance(obj, xr.DataArray):
        name = obj.name or "data"
        ds = obj.rename(name).to_dataset()
    else:
        ds = obj.copy()

    for name in list(ds.coords):
        var = ds.coords[name]
        if var.dtype.kind == "O":
            if name in ds.dims:
                # Dimension coordinates must be retained. These should normally
                # be strings if they arrive as dtype=object.
                try:
                    ds = ds.assign_coords({name: var.astype(str)})
                except Exception as exc:
                    raise TypeError(
                        f"Cannot serialise object-dtype dimension coordinate {name!r}"
                    ) from exc
            else:
                print(
                    f"Dropping non-NetCDF auxiliary coordinate {name!r} "
                    f"(dtype=object)"
                )
                ds = ds.drop_vars(name)

    # Object-dtype data variables should not occur in the Coastlines caches.
    # Fail explicitly rather than silently corrupting scientific data.
    bad_vars = [
        name for name, var in ds.data_vars.items()
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
    return all((folder / f"{year}_{v}{suffix}.tif").exists() for v in variables)


def versioned_paths(site, raster_version=None, vector_version=None):
    """Return all version-specific paths for one AOI."""
    raster_version = raster_version or RASTER_VERSION
    vector_version = vector_version or VECTOR_VERSION
    cache = CACHE_ROOT / raster_version / f"{site}_{raster_version}"
    raster_dir = RASTER_ROOT / raster_version / f"{site}_{raster_version}"
    vector_dir = VECTOR_ROOT / vector_version / f"{site}_{vector_version}"

    return cache, raster_dir, vector_dir


def clear_current_version(site, raster_version=None, vector_version=None):
    """Delete only this AOI's products for the current versions."""
    cache, raster_dir, vector_dir = versioned_paths(site, raster_version, vector_version)

    for path in [cache, raster_dir, vector_dir]:
        if path.exists():
            print(f"Removing for fresh run: {path}")
            shutil.rmtree(path)


def ocean_seeds(yearly_ds, max_points=50):
    """Use persistently wet raster-edge pixels as ocean seed points."""
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
        raise RuntimeError("No persistent-water pixels found on the raster edge")

    if len(candidates) > max_points:
        candidates = candidates[
            np.linspace(0, len(candidates) - 1, max_points).astype(int)
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
# ONE AOI
# =============================================================================


def run_aoi(row, fresh=False, *, sensor="landsat", years=None, tide_model=None):
    """Run the complete raster-to-shoreline workflow for one AOI."""

    name = str(row[NAME_FIELD])
    site = slugify(name)
    geometry = row.geometry
    tide_model = tide_model or TIDE_MODEL
    if sensor == "sentinel2":
        start_year, end_year = years or SENTINEL2_YEARS
        cutoff_start, cutoff_end = start_year - 1, end_year + 1
        resolution = 20
        suffix = f"_sentinel2_{start_year}_{end_year}"
        loader = load_sentinel2_indices_stac
        options = {}
    else:
        start_year, end_year = START_YEAR, END_YEAR
        cutoff_start, cutoff_end = CUTOFF_START, CUTOFF_END
        resolution = RESOLUTION
        suffix = ""
        loader = load_water_index_stac
        options = {"platforms": PLATFORMS}
    model_cache_name = TIDE_MODEL
    if tide_model != TIDE_MODEL:
        model_cache_name = slugify(tide_model)
        suffix += f"_{model_cache_name}"
    raster_version = RASTER_VERSION + suffix
    vector_version = VECTOR_VERSION + suffix

    bbox = list(
        gpd.GeoSeries([geometry], crs=CRS)
        .to_crs("EPSG:4326")
        .total_bounds
    )

    print("\n" + "=" * 78)
    print(f"AOI:            {name}")
    print(f"Site ID:        {site}")
    print(f"Raster version: {raster_version}")
    print(f"Vector version: {vector_version}")
    print("=" * 78)

    if fresh:
        clear_current_version(site, raster_version, vector_version)

    cache, raster_dir, vector_dir = versioned_paths(site, raster_version, vector_version)

    dem_file = cache / "cop_dem_glo30_bng.tif"
    cutoff_file = cache / f"{model_cache_name}_cutoffs_{cutoff_start}_{cutoff_end}.nc"
    tide_dir = cache / "tides"
    obs_dir = cache / "tidal_observations"

    for folder in [cache, tide_dir, obs_dir, raster_dir, vector_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    output_geojson = (
        vector_dir
        / f"annualshorelines_{site}_{vector_version}_{WATER_INDEX}_{INDEX_THRESHOLD:.2f}.geojson"
    )
    output_gpkg = (
        vector_dir
        / f"annualshorelines_{site}_{vector_version}_{WATER_INDEX}_{INDEX_THRESHOLD:.2f}_BNG.gpkg"
    )

    # Resumable mode: a completed vector product means this AOI is finished.
    if output_geojson.exists() and not fresh:
        print(f"Already complete: {output_geojson}")
        return

    # -------------------------------------------------------------------------
    # STAGE 1: DEM + Landsat loader
    # -------------------------------------------------------------------------

    if dem_file.exists():
        print("Using cached Copernicus GLO-30 DEM")
        dem = (
            rioxarray.open_rasterio(dem_file)
            .squeeze("band", drop=True)
            .load()
        )
    else:
        print("Loading Copernicus GLO-30 DEM...")
        dem = load_dem_stac(
            bbox,
            output_crs=CRS,
            resolution=resolution,
        ).compute()
        dem.rio.to_raster(dem_file)

    def load_year(year, shadow=False, fail_on_error=True):
        return loader(
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            output_crs=CRS,
            resolution=resolution,
            cloud_cover=CLOUD_COVER,
            fail_on_error=fail_on_error,
            dem=dem if shadow else None,
            mask_terrain_shadow=shadow,
            terrain_shadow_threshold=SHADOW_THRESHOLD,
            terrain_shadow_radius=SHADOW_RADIUS,
            **options,
        )

    # -------------------------------------------------------------------------
    # STAGE 2: site-specific tidal cutoffs
    # -------------------------------------------------------------------------

    if cutoff_file.exists():
        print("Using cached site-specific tidal cutoffs")
        with xr.open_dataset(cutoff_file) as ds:
            cutoff_min = ds.tide_cutoff_min.load()
            cutoff_max = ds.tide_cutoff_max.load()
    else:
        print("Building site-specific tidal cutoffs...")

        record = loader(
            bbox=bbox,
            datetime=f"{cutoff_start}-01-01/{cutoff_end}-12-31",
            output_crs=CRS,
            resolution=resolution,
            cloud_cover=CLOUD_COVER,
            **options,
        )

        tides_lowres = pixel_tides(
            data=record,
            model=tide_model,
            directory=TIDE_MODEL_DIR,
            resample=False,
        ).compute()

        cutoff_min, cutoff_max = tide_cutoffs(
            ds=load_year(start_year),
            tides_lowres=tides_lowres,
            tide_centre=0.0,
        )

        cutoff_ds = xr.Dataset(
            {
                "tide_cutoff_min": cutoff_min,
                "tide_cutoff_max": cutoff_max,
            }
        )
        save_netcdf_safe(cutoff_ds, cutoff_file)

    # -------------------------------------------------------------------------
    # STAGE 3: one year of tide-filtered observations
    # -------------------------------------------------------------------------

    def observations(year):
        obs_file = obs_dir / f"{year}_tidal.nc"
        tide_file = tide_dir / f"{year}_tides.nc"

        if obs_file.exists():
            with xr.open_dataset(obs_file) as ds:
                return ds.load()

        if tide_file.exists():
            with xr.open_dataset(tide_file) as ds:
                tides = ds.tide_m.load()
        else:
            tides = pixel_tides(
                data=load_year(year),
                model=tide_model,
                directory=TIDE_MODEL_DIR,
                resample=True,
            ).rename("tide_m").compute()

            save_netcdf_safe(tides.to_dataset(), tide_file)

        for attempt in range(1, 4):
            pc_sas.TOKEN_CACHE.clear()

            try:
                ds = load_year(
                    year,
                    shadow=True,
                    fail_on_error=(attempt < 3),
                )

                # The current catalogue may contain observations that were not
                # present when a cached tide file was created. Rebuild if needed.
                if not set(ds.time.values).issubset(set(tides.time.values)):
                    tides = pixel_tides(
                        data=ds,
                        model=tide_model,
                        directory=TIDE_MODEL_DIR,
                        resample=True,
                    ).rename("tide_m").compute()

                    save_netcdf_safe(tides.to_dataset(), tide_file)

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

    previous = observations(start_year - 1)
    current = observations(start_year)
    future = observations(start_year + 1)

    for year in range(start_year, end_year + 1):
        if not rasters_exist(raster_dir, year):
            tidal_composite(
                current,
                year,
                "year",
                str(raster_dir),
                export_geotiff=True,
            )

        if not rasters_exist(raster_dir, year, "_gapfill"):
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

        if year < end_year:
            previous = current
            current = future
            future = observations(year + 2)

    # -------------------------------------------------------------------------
    # STAGE 5: DEA masking + sub-pixel shoreline extraction
    # -------------------------------------------------------------------------

    print("Extracting annual MSL shorelines...")

    # Use the native DEA versioned raster layout:
    # outputs/dea_rasters/<RASTER_VERSION>/<site>_<RASTER_VERSION>/...
    yearly_ds, gapfill_ds = load_rasters(
        path=str(RASTER_ROOT),
        raster_version=raster_version,
        study_area=site,
        water_index=WATER_INDEX,
        start_year=start_year,
        end_year=end_year,
    )

    seeds = ocean_seeds(yearly_ds)
    seeds.to_file(
        vector_dir / "ocean_seed_points.gpkg",
        driver="GPKG",
    )

    masked_ds, certainty = contours_preprocess(
        yearly_ds=yearly_ds,
        gapfill_ds=gapfill_ds,
        water_index=WATER_INDEX,
        index_threshold=INDEX_THRESHOLD,
        tide_points_gdf=seeds,
        buffer_pixels=round(33 * RESOLUTION / resolution),
        mask_landcover=False,
        mask_ndwi=True,
        mask_temporal=start_year < end_year,
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
        {str(y): m for y, m in certainty.items()},
    )
    contours.index = contours.index.astype(int)
    contours.index.name = "year"

    shorelines = contours.reset_index()
    shorelines = shorelines[
        shorelines.geometry.notnull()
        & ~shorelines.geometry.is_empty
    ].copy()

    shorelines = shorelines.to_crs(CRS)
    shorelines["geometry"] = shorelines.geometry.intersection(geometry)
    shorelines = shorelines[~shorelines.geometry.is_empty].copy()

    # Provenance / version metadata
    shorelines["indicator"] = "DEA_MSL"
    shorelines["water_index"] = WATER_INDEX
    shorelines["threshold"] = INDEX_THRESHOLD
    shorelines["tide_model"] = tide_model
    shorelines["tide_datum"] = "model MSL"
    shorelines["raster_version"] = raster_version
    shorelines["vector_version"] = vector_version
    if sensor == "sentinel2":
        shorelines["sensor"] = "sentinel2"

    shorelines["segment_no"] = shorelines.groupby("year").cumcount() + 1
    shorelines["shoreline_id"] = (
        shorelines.year.astype(str)
        + "_"
        + shorelines.segment_no.astype(str).str.zfill(3)
    )

    # BNG working product + WGS84 handoff product
    shorelines.to_file(
        output_gpkg,
        driver="GPKG",
    )

    shorelines.to_crs("EPSG:4326").to_file(
        output_geojson,
        driver="GeoJSON",
    )

    print(f"Finished {name}")
    print(f"Raster directory: {raster_dir}")
    print(f"Vector output:    {output_geojson}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    args = parse_args()

    if not AOI_FILE.exists():
        raise FileNotFoundError(f"AOI file not found: {AOI_FILE}")

    if not TIDE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Tide model directory not found: {TIDE_MODEL_DIR}"
        )

    aois = gpd.read_file(AOI_FILE)

    if aois.crs is None:
        raise ValueError("AOI file has no CRS")

    if NAME_FIELD not in aois.columns:
        raise ValueError(
            f"AOI file has no '{NAME_FIELD}' field"
        )

    aois = aois.to_crs(CRS)

    sites = args.sites if args.sites is not None else SITES
    if sites is not None:
        wanted = {site.casefold() for site in sites}
        aois = aois[
            aois[NAME_FIELD]
            .astype(str)
            .str.casefold()
            .isin(wanted)
        ]

    print()
    print("=" * 78)
    print("SCOTLAND DEA COASTLINES AOI RUN")
    print("=" * 78)
    print(f"AOIs:           {len(aois)}")
    print(f"Fresh run:      {args.fresh}")
    years = args.years or (SENTINEL2_YEARS if args.sensor == "sentinel2" else (START_YEAR, END_YEAR))
    print(f"Sensor:         {args.sensor}")
    print(f"Years:          {years[0]}-{years[1]}")
    print(f"Tide model:     {args.tide_model}")
    print("=" * 78)

    for _, row in aois.iterrows():
        try:
            run_aoi(row, fresh=args.fresh, sensor=args.sensor, years=args.years,
                    tide_model=args.tide_model)
        except Exception as error:
            print(f"ERROR: {row[NAME_FIELD]} failed: {error}")


if __name__ == "__main__":
    main()
