#!/usr/bin/env python
"""
Full Montrose DEA Coastlines tide-model sensitivity experiment.

Compares EOT20 and FES2022 while keeping the Landsat, compositing,
shoreline extraction and DEA rate methodology fixed.

Default behaviour:
  * reuses the existing validated EOT20 raster products/cache where possible
  * builds a completely separate FES2022 cutoff/cache/raster stack
  * extracts attributed annual shorelines for both models
  * calculates rates at one common set of 30 m baseline points
  * exports rate differences and QA summaries/figures

Run from the repository root, e.g.:

    conda activate DEA_Coastlines
    python -u scripts/run_montrose_tide_model_comparison.py

For the strictest possible experiment, rebuild EOT20 with the current STAC
catalogue too:

    python -u scripts/run_montrose_tide_model_comparison.py --fresh-eot20

The script is resumable: model-specific tide caches, filtered-observation
caches and completed annual/gapfill rasters are reused automatically.
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer.sas as pc_sas
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import linregress
from shapely.geometry import Point
from shapely.ops import nearest_points

from dea_tools.spatial import subpixel_contours
from eo_tides.eo import pixel_tides

from coastlines.raster import (
    load_tidal_subset,
    tidal_composite,
    tide_cutoffs,
)
from coastlines.stac import load_water_index_stac
from coastlines.vector import (
    annual_movements,
    calculate_regressions,
    contour_certainty,
    contours_preprocess,
    load_rasters,
    points_on_line,
)


# =====================================================================
# FIXED EXPERIMENT SETTINGS
# =====================================================================

BBOX = [-2.52, 56.70, -2.42, 56.76]

START_YEAR = 1988
END_YEAR = 2025
CONTEXT_START_YEAR = START_YEAR - 1
CONTEXT_END_YEAR = END_YEAR + 1

# Long-term tide range used by the DEA middle-50% tidal filter.
CUTOFF_START_YEAR = 1984
CUTOFF_END_YEAR = 2025

BASELINE_YEAR = 2020
POINT_SPACING_M = 30

WATER_INDEX = "mndwi"
INDEX_THRESHOLD = 0.0
STUDY_AREA = "montrose"

PLATFORMS = [
    "landsat-5",
    "landsat-7",
    "landsat-8",
    "landsat-9",
]

RASTER_ROOT = Path("outputs/dea_rasters")
COMPARISON_ROOT = Path("outputs/tide_model_comparison")

MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5


# =====================================================================
# COMMAND LINE
# =====================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Montrose DEA Coastlines EOT20 versus FES2022 "
            "tide-model sensitivity experiment."
        )
    )

    parser.add_argument(
        "--tide-model-dir",
        default="/home/mh322u/tide_models_scotland",
        help=(
            "Root directory recognised by eo-tides containing EOT20 and "
            "the cropped FES2022 model."
        ),
    )

    parser.add_argument(
        "--fresh-eot20",
        action="store_true",
        help=(
            "Rebuild EOT20 into a comparison-specific cache/raster version "
            "instead of reusing outputs/dea_rasters/scotland_v0.1 and "
            "outputs/montrose_dea_cache."
        ),
    )

    parser.add_argument(
        "--skip-raster-generation",
        action="store_true",
        help="Skip cutoff/cache/raster generation and use existing products.",
    )

    parser.add_argument(
        "--allow-scene-mismatch",
        action="store_true",
        help=(
            "Continue to rate comparison even if EOT20 and FES2022 source "
            "Landsat timestamp sets differ. This is not recommended."
        ),
    )

    return parser.parse_args()


# =====================================================================
# GENERAL HELPERS
# =====================================================================


def heading(text, width=78):
    print()
    print("=" * width)
    print(text)
    print("=" * width)


def ensure_dirs():
    for path in [
        COMPARISON_ROOT,
        COMPARISON_ROOT / "cutoffs",
        COMPARISON_ROOT / "shorelines",
        COMPARISON_ROOT / "rates",
        COMPARISON_ROOT / "qa",
        COMPARISON_ROOT / "summaries",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def experiment_configs(args):
    """Return model-specific cache/raster/cutoff settings."""

    if args.fresh_eot20:
        eot20_raster_version = "scotland_eot20_compare_v0.1"
        eot20_cache_root = Path("outputs/montrose_dea_cache/EOT20_compare")
        eot20_cutoff = (
            COMPARISON_ROOT
            / "cutoffs"
            / f"montrose_EOT20_cutoffs_{CUTOFF_START_YEAR}_{CUTOFF_END_YEAR}.nc"
        )
    else:
        # Reuse the validated run already present in this repository workflow.
        eot20_raster_version = "scotland_v0.1"
        eot20_cache_root = Path("outputs/montrose_dea_cache")
        eot20_cutoff = Path(
            f"montrose_tidal_cutoffs_{CUTOFF_START_YEAR}_{CUTOFF_END_YEAR}.nc"
        )

    return {
        "EOT20": {
            "model": "EOT20",
            "tide_model_dir": Path(args.tide_model_dir),
            "raster_version": eot20_raster_version,
            "cache_root": eot20_cache_root,
            "cutoff_file": eot20_cutoff,
        },
        "FES2022": {
            "model": "FES2022",
            "tide_model_dir": Path(args.tide_model_dir),
            "raster_version": "scotland_fes2022_v0.1",
            "cache_root": Path("outputs/montrose_dea_cache/FES2022"),
            "cutoff_file": (
                COMPARISON_ROOT
                / "cutoffs"
                / f"montrose_FES2022_cutoffs_{CUTOFF_START_YEAR}_{CUTOFF_END_YEAR}.nc"
            ),
        },
    }


def raster_output_dir(config):
    version = config["raster_version"]
    return RASTER_ROOT / version / f"{STUDY_AREA}_{version}"


def stac_load_year(year, fail_on_error=True):
    """Create a lazy annual Landsat water-index dataset."""

    kwargs = {
        "bbox": BBOX,
        "datetime": f"{year}-01-01/{year}-12-31",
        "platforms": PLATFORMS,
    }

    signature = inspect.signature(load_water_index_stac)
    if "fail_on_error" in signature.parameters:
        kwargs["fail_on_error"] = fail_on_error

    return load_water_index_stac(**kwargs)


def stac_load_period(start_year, end_year):
    """Create a lazy Landsat dataset spanning a multi-year period."""

    return load_water_index_stac(
        bbox=BBOX,
        datetime=f"{start_year}-01-01/{end_year}-12-31",
        platforms=PLATFORMS,
    )


# =====================================================================
# MODEL-SPECIFIC LONG-TERM TIDAL CUTOFFS
# =====================================================================


def build_or_load_cutoffs(config):
    cutoff_file = Path(config["cutoff_file"])

    if cutoff_file.exists():
        print(f"Using existing {config['model']} cutoffs: {cutoff_file}")
        with xr.open_dataset(cutoff_file) as ds:
            cutoff_min = ds["tide_cutoff_min"].load()
            cutoff_max = ds["tide_cutoff_max"].load()
        return cutoff_min, cutoff_max

    heading(
        f"{config['model']}: CALCULATING {CUTOFF_START_YEAR}-{CUTOFF_END_YEAR} "
        "TIDAL CUTOFFS"
    )

    # Full record supplies acquisition times and the long-term tidal range.
    ds_full = stac_load_period(CUTOFF_START_YEAR, CUTOFF_END_YEAR)

    print(
        f"Full Landsat record contains {ds_full.sizes['time']} acquisition times."
    )

    print(f"Modelling low-resolution {config['model']} tides...")
    tides_lowres = pixel_tides(
        data=ds_full,
        model=config["model"],
        directory=str(config["tide_model_dir"]),
        resample=False,
    ).compute()

    # Use one normal annual Landsat stack only to define the target 30 m grid.
    ds_like = stac_load_year(BASELINE_YEAR)

    cutoff_min, cutoff_max = tide_cutoffs(
        ds=ds_like,
        tides_lowres=tides_lowres,
        tide_centre=0.0,
    )

    cutoff_min = cutoff_min.compute()
    cutoff_max = cutoff_max.compute()

    cutoff_file.parent.mkdir(parents=True, exist_ok=True)

    cutoff_ds = xr.Dataset(
        {
            "tide_cutoff_min": cutoff_min,
            "tide_cutoff_max": cutoff_max,
        },
        attrs={
            "tide_model": config["model"],
            "cutoff_start_year": CUTOFF_START_YEAR,
            "cutoff_end_year": CUTOFF_END_YEAR,
            "tide_centre_m": 0.0,
            "method": "DEA Coastlines middle 50 percent satellite-observed tidal range",
        },
    )

    cutoff_ds.to_netcdf(cutoff_file)

    print(f"Saved {config['model']} cutoffs: {cutoff_file}")
    print(
        f"Mean cutoff window: {float(cutoff_min.mean()):.3f} to "
        f"{float(cutoff_max.mean()):.3f} m"
    )

    return cutoff_min, cutoff_max


# =====================================================================
# MODEL-SPECIFIC TIDE AND FILTERED-OBSERVATION CACHES
# =====================================================================


def cache_dirs(config):
    tide_dir = Path(config["cache_root"]) / "tides"
    obs_dir = Path(config["cache_root"]) / "tidal_observations"
    tide_dir.mkdir(parents=True, exist_ok=True)
    obs_dir.mkdir(parents=True, exist_ok=True)
    return tide_dir, obs_dir


def save_tide_cache(tides, tide_file, model):
    tides = tides.rename("tide_m")
    ds = tides.to_dataset()
    ds.attrs["tide_model"] = model
    ds.to_netcdf(tide_file)
    print(f"Saved tide cache: {tide_file}")


def load_tide_cache(tide_file):
    with xr.open_dataset(tide_file) as ds:
        return ds["tide_m"].load()


def build_tides_for_year(year, config):
    tide_cache_dir, _ = cache_dirs(config)
    tide_file = tide_cache_dir / f"{year}_tides.nc"

    if tide_file.exists():
        print(f"Using cached {config['model']} tides: {tide_file}")
        return load_tide_cache(tide_file)

    print(f"{year}: querying Landsat acquisition times for {config['model']}...")
    ds_meta = stac_load_year(year)
    print(f"{year}: {ds_meta.sizes['time']} Landsat observations")

    print(f"{year}: modelling {config['model']} pixel tides...")
    tides = pixel_tides(
        data=ds_meta,
        model=config["model"],
        directory=str(config["tide_model_dir"]),
        resample=True,
    ).rename("tide_m")

    tides = tides.compute()

    valid_fraction = float(np.isfinite(tides.values).mean())
    print(f"{year}: {config['model']} tide-grid valid fraction = {valid_fraction:.3f}")

    if valid_fraction < 0.50:
        print(
            f"WARNING: only {valid_fraction:.1%} of modelled tide values are valid. "
            "Standard FES2022 may have limited land/coastal coverage; inspect this "
            "before interpreting shoreline differences."
        )

    save_tide_cache(tides, tide_file, config["model"])
    return tides


def tide_times_match(tides, ds):
    tide_times = set(tides.time.values.astype("datetime64[ns]"))
    image_times = set(ds.time.values.astype("datetime64[ns]"))
    return image_times.issubset(tide_times)


def regenerate_tides(year, ds, config):
    print(
        f"{year}: Landsat timestamps changed; regenerating {config['model']} tide cache."
    )

    tides = pixel_tides(
        data=ds,
        model=config["model"],
        directory=str(config["tide_model_dir"]),
        resample=True,
    ).rename("tide_m").compute()

    tide_cache_dir, _ = cache_dirs(config)
    tide_file = tide_cache_dir / f"{year}_tides.nc"
    save_tide_cache(tides, tide_file, config["model"])
    return tides


def build_filtered_observations(year, config, cutoff_min, cutoff_max):
    _, observation_cache_dir = cache_dirs(config)
    observation_file = observation_cache_dir / f"{year}_tidal.nc"

    if observation_file.exists():
        with xr.open_dataset(observation_file) as cached:
            required = {"mndwi", "ndwi", "tide_m"}
            if required.issubset(set(cached.data_vars)):
                print(
                    f"Using cached {config['model']} tide-filtered observations: "
                    f"{observation_file}"
                )
                return cached.load()

        print(f"{year}: incomplete observation cache; rebuilding {observation_file}")
        observation_file.unlink()

    tides = build_tides_for_year(year, config)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(
            f"{year}: {config['model']} raster download attempt "
            f"{attempt}/{MAX_ATTEMPTS}"
        )

        pc_sas.TOKEN_CACHE.clear()
        strict = attempt < MAX_ATTEMPTS

        try:
            ds = stac_load_year(year, fail_on_error=strict)

            if not tide_times_match(tides, ds):
                tides = regenerate_tides(year, ds, config)

            tides_year = tides.sel(time=ds.time)

            ds = ds[["mndwi", "ndwi"]]
            ds["tide_m"] = tides_year

            target = tides_year.isel(time=0)
            local_min = cutoff_min.interp_like(target, method="linear")
            local_max = cutoff_max.interp_like(target, method="linear")

            print(
                f"{year}: applying {config['model']} fixed "
                f"{CUTOFF_START_YEAR}-{CUTOFF_END_YEAR} tidal filter..."
            )

            tidal_ds = load_tidal_subset(
                year_ds=ds,
                tide_cutoff_min=local_min,
                tide_cutoff_max=local_max,
            )

            tidal_ds.attrs["tide_model"] = config["model"]
            tidal_ds.attrs["source_scene_count"] = int(ds.sizes["time"])
            tidal_ds.attrs["retained_timestep_count"] = int(tidal_ds.sizes["time"])

            tidal_ds.to_netcdf(observation_file)

            print(
                f"{year}: retained {tidal_ds.sizes['time']} timesteps; "
                f"saved {observation_file}"
            )

            return tidal_ds

        except Exception as error:
            print(f"{year}: raster read failed: {error}")
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"Waiting {RETRY_WAIT_SECONDS} s before retry...")
            time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Unable to process {config['model']} {year}")


# =====================================================================
# MODEL-SPECIFIC ANNUAL + GAPFILL RASTERS
# =====================================================================


def annual_files_exist(year, output_dir):
    required = [
        output_dir / f"{year}_mndwi.tif",
        output_dir / f"{year}_ndwi.tif",
        output_dir / f"{year}_count.tif",
        output_dir / f"{year}_stdev.tif",
        output_dir / f"{year}_tide_m.tif",
    ]
    return all(path.exists() for path in required)


def gapfill_files_exist(year, output_dir):
    required = [
        output_dir / f"{year}_mndwi_gapfill.tif",
        output_dir / f"{year}_ndwi_gapfill.tif",
        output_dir / f"{year}_count_gapfill.tif",
        output_dir / f"{year}_stdev_gapfill.tif",
        output_dir / f"{year}_tide_m_gapfill.tif",
    ]
    return all(path.exists() for path in required)


def build_model_rasters(config):
    heading(f"BUILDING / CHECKING {config['model']} DEA RASTERS")

    output_dir = raster_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    cutoff_min, cutoff_max = build_or_load_cutoffs(config)

    # Stage 1: ensure model-specific filtered observation caches exist.
    for year in range(CONTEXT_START_YEAR, CONTEXT_END_YEAR + 1):
        print()
        print("-" * 78)
        print(f"{config['model']}: CACHING {year}")
        print("-" * 78)
        build_filtered_observations(year, config, cutoff_min, cutoff_max)

    # Stage 2: annual and true 3-year observation-level gapfill composites.
    previous_ds = build_filtered_observations(
        START_YEAR - 1, config, cutoff_min, cutoff_max
    )
    current_ds = build_filtered_observations(
        START_YEAR, config, cutoff_min, cutoff_max
    )
    future_ds = build_filtered_observations(
        START_YEAR + 1, config, cutoff_min, cutoff_max
    )

    for year in range(START_YEAR, END_YEAR + 1):
        print()
        print("-" * 78)
        print(f"{config['model']}: COMPOSITING {year}")
        print("-" * 78)

        if annual_files_exist(year, output_dir):
            print(f"{year}: annual rasters already exist; skipping.")
        else:
            tidal_composite(
                year_ds=current_ds,
                label=year,
                label_dim="year",
                output_dir=str(output_dir),
                export_geotiff=True,
            )

        if gapfill_files_exist(year, output_dir):
            print(f"{year}: gapfill rasters already exist; skipping.")
        else:
            gapfill_ds = xr.concat(
                [previous_ds, current_ds, future_ds],
                dim="time",
            ).sortby("time")

            tidal_composite(
                year_ds=gapfill_ds,
                label=year,
                label_dim="year",
                output_dir=str(output_dir),
                output_suffix="_gapfill",
                export_geotiff=True,
            )

            del gapfill_ds

        if year < END_YEAR:
            previous_ds = current_ds
            current_ds = future_ds
            future_ds = build_filtered_observations(
                year + 2, config, cutoff_min, cutoff_max
            )

    print(f"{config['model']} raster stack ready: {output_dir}")


# =====================================================================
# SHORELINE EXTRACTION + CERTAINTY ATTRIBUTION
# =====================================================================


def create_ocean_seed_points(yearly_ds):
    x_values = yearly_ds.x.values
    y_values = yearly_ds.y.values
    ocean_x = float(x_values[-3])
    indices = np.linspace(0, len(y_values) - 1, 12).astype(int)

    points = [Point(ocean_x, float(y_values[i])) for i in indices]

    crs = yearly_ds.rio.crs
    if crs is None:
        raise ValueError("Raster stack has no CRS.")

    return gpd.GeoDataFrame(geometry=points, crs=crs)


def extract_model_shorelines(config):
    heading(f"{config['model']}: EXTRACTING ATTRIBUTED ANNUAL SHORELINES")

    yearly_ds, gapfill_ds = load_rasters(
        path=str(RASTER_ROOT),
        raster_version=config["raster_version"],
        study_area=STUDY_AREA,
        water_index=WATER_INDEX,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    tide_points_gdf = create_ocean_seed_points(yearly_ds)

    masked_ds, certainty_masks = contours_preprocess(
        yearly_ds=yearly_ds,
        gapfill_ds=gapfill_ds,
        water_index=WATER_INDEX,
        index_threshold=INDEX_THRESHOLD,
        tide_points_gdf=tide_points_gdf,
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

    # contour_certainty expects matching string year keys in this DE Africa fork.
    contours.index = contours.index.astype(str)
    certainty_masks = {str(year): mask for year, mask in certainty_masks.items()}
    contours = contour_certainty(contours, certainty_masks)

    contours.index = contours.index.astype(int)
    contours.index.name = "year"
    contours = contours[
        contours.geometry.notnull() & ~contours.geometry.is_empty
    ].copy()

    contours["indicator"] = "DEA_MSL"
    contours["water_index"] = WATER_INDEX
    contours["threshold"] = float(INDEX_THRESHOLD)
    contours["tide_model"] = config["model"]
    contours["tide_datum"] = "model MSL"
    contours["method"] = "DE Africa Coastlines subpixel contour"
    contours["raster_version"] = config["raster_version"]

    shorelines = contours.reset_index()
    shorelines["segment_no"] = shorelines.groupby("year").cumcount() + 1
    shorelines["shoreline_id"] = (
        shorelines["year"].astype(str)
        + "_"
        + shorelines["segment_no"].astype(str).str.zfill(3)
    )

    keep = [
        "shoreline_id",
        "year",
        "segment_no",
        "indicator",
        "certainty",
        "water_index",
        "threshold",
        "tide_model",
        "tide_datum",
        "method",
        "raster_version",
        "geometry",
    ]
    shorelines = shorelines[[c for c in keep if c in shorelines.columns]]

    output_file = (
        COMPARISON_ROOT
        / "shorelines"
        / f"montrose_{config['model']}_DEA_MSL_{START_YEAR}_{END_YEAR}.geojson"
    )

    shorelines.to_crs("EPSG:4326").to_file(
        output_file,
        driver="GeoJSON",
        index=False,
    )

    summary = (
        shorelines.assign(length_m=shorelines.geometry.length)
        .groupby(["year", "certainty"], dropna=False)
        .agg(n_segments=("shoreline_id", "count"), length_m=("length_m", "sum"))
        .reset_index()
    )
    summary.to_csv(
        COMPARISON_ROOT
        / "summaries"
        / f"montrose_{config['model']}_shoreline_certainty.csv",
        index=False,
    )

    print(f"Saved {config['model']} shorelines: {output_file}")
    return output_file, yearly_ds


# =====================================================================
# SOURCE-SCENE AND ACCEPTED-OBSERVATION COMPARISON
# =====================================================================


def nc_time_set(path):
    with xr.open_dataset(path) as ds:
        return set(ds.time.values.astype("datetime64[ns]"))


def observation_comparison(configs, yearly_datasets):
    heading("COMPARING SOURCE SCENES AND TIDAL ACCEPTANCE")

    rows = []
    any_source_mismatch = False

    for year in range(START_YEAR, END_YEAR + 1):
        model_data = {}

        for model, config in configs.items():
            tide_dir, obs_dir = cache_dirs(config)
            tide_file = tide_dir / f"{year}_tides.nc"
            obs_file = obs_dir / f"{year}_tidal.nc"

            if not tide_file.exists() or not obs_file.exists():
                raise FileNotFoundError(
                    f"Missing cache required for comparison: {tide_file} / {obs_file}"
                )

            source_times = nc_time_set(tide_file)
            accepted_times = nc_time_set(obs_file)

            mean_count = float(
                yearly_datasets[model]["count"].sel(year=year).mean(skipna=True).values
            )

            model_data[model] = {
                "source": source_times,
                "accepted": accepted_times,
                "mean_count": mean_count,
            }

        e = model_data["EOT20"]
        f = model_data["FES2022"]
        source_match = e["source"] == f["source"]
        any_source_mismatch |= not source_match

        rows.append(
            {
                "year": year,
                "source_scenes_eot20": len(e["source"]),
                "source_scenes_fes2022": len(f["source"]),
                "source_scene_sets_identical": source_match,
                "accepted_images_eot20": len(e["accepted"]),
                "accepted_images_fes2022": len(f["accepted"]),
                "accepted_common": len(e["accepted"] & f["accepted"]),
                "accepted_only_eot20": len(e["accepted"] - f["accepted"]),
                "accepted_only_fes2022": len(f["accepted"] - e["accepted"]),
                "mean_pixel_count_eot20": e["mean_count"],
                "mean_pixel_count_fes2022": f["mean_count"],
            }
        )

    df = pd.DataFrame(rows)
    out_csv = COMPARISON_ROOT / "summaries" / "montrose_observation_comparison.csv"
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(df.year, df.accepted_images_eot20, label="EOT20", linewidth=1.4)
    axes[0].plot(df.year, df.accepted_images_fes2022, label="FES2022", linewidth=1.4)
    axes[0].set_ylabel("Accepted images")
    axes[0].legend()

    axes[1].plot(df.year, df.mean_pixel_count_eot20, label="EOT20", linewidth=1.4)
    axes[1].plot(df.year, df.mean_pixel_count_fes2022, label="FES2022", linewidth=1.4)
    axes[1].set_ylabel("Mean accepted\nobservations per pixel")
    axes[1].set_xlabel("Year")

    fig.suptitle("Montrose tidal-filter acceptance: EOT20 vs FES2022")
    fig.tight_layout()
    fig.savefig(
        COMPARISON_ROOT / "qa" / "montrose_observation_acceptance_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    return df, any_source_mismatch


# =====================================================================
# DEA RATE COMPARISON AT ONE COMMON SET OF 30 m POINTS
# =====================================================================


def load_dissolved_contours(path, target_crs):
    gdf = gpd.read_file(path).to_crs(target_crs)
    gdf["year"] = gdf["year"].astype(int)
    dissolved = gdf[["year", "geometry"]].dissolve(by="year")
    dissolved.index = dissolved.index.astype(str)
    dissolved.index.name = "year"
    return dissolved


def restore_actual_baseline_distance(
    movement_gdf,
    common_points,
    contours_gdf,
    yearly_ds,
    baseline_year,
):
    """
    annual_movements() forces dist_<baseline_year> to zero. That is correct
    when the rate points were generated from that model's baseline shoreline,
    but our FES2022 calculation deliberately uses EOT20-derived common points.

    Restore the actual model-specific signed baseline-year distance so both
    regressions use the same fixed geometry without artificially pinning
    FES2022 to zero in one year.
    """

    year_str = str(baseline_year)
    contour = contours_gdf.loc[[year_str]].geometry.iloc[0]
    array = yearly_ds[WATER_INDEX].sel(year=baseline_year)

    comparison_points = [nearest_points(p, contour)[1] for p in common_points.geometry]

    p1x = xr.DataArray(common_points.geometry.x.to_numpy(), dims="z")
    p1y = xr.DataArray(common_points.geometry.y.to_numpy(), dims="z")
    p2x = xr.DataArray(np.array([p.x for p in comparison_points]), dims="z")
    p2y = xr.DataArray(np.array([p.y for p in comparison_points]), dims="z")

    index_at_common = array.interp(x=p1x, y=p1y)
    index_at_model_shore = array.interp(x=p2x, y=p2y)

    distances = np.array(
        [p.distance(q) for p, q in zip(common_points.geometry, comparison_points)],
        dtype=float,
    )

    # Same directional rule used by annual_movements().
    signs = np.where(index_at_model_shore.values > index_at_common.values, 1.0, -1.0)
    invalid = ~np.isfinite(index_at_common.values) | ~np.isfinite(index_at_model_shore.values)
    signed = distances * signs
    signed[invalid] = np.nan

    movement_gdf[f"dist_{baseline_year}"] = signed
    return movement_gdf


def calculate_common_point_rates(
    configs,
    shoreline_files,
    yearly_datasets,
):
    heading("CALCULATING EOT20 AND FES2022 RATES AT COMMON 30 m POINTS")

    crs = yearly_datasets["EOT20"].rio.crs
    if crs is None:
        raise ValueError("EOT20 raster stack has no CRS.")

    contours = {
        model: load_dissolved_contours(path, crs)
        for model, path in shoreline_files.items()
    }

    common_points = points_on_line(
        contours["EOT20"],
        str(BASELINE_YEAR),
        distance=POINT_SPACING_M,
    )
    common_points["point_id"] = np.arange(len(common_points), dtype=int)
    common_points = common_points.set_index("point_id", drop=False)
    common_points["alongshore_m"] = common_points["point_id"] * POINT_SPACING_M

    rate_outputs = {}

    for model in ["EOT20", "FES2022"]:
        print(f"Calculating {model} annual movements...")

        movement = annual_movements(
            points_gdf=common_points.copy(),
            contours_gdf=contours[model],
            yearly_ds=yearly_datasets[model],
            baseline_year=str(BASELINE_YEAR),
            water_index=WATER_INDEX,
            max_valid_dist=5000,
        )

        movement = restore_actual_baseline_distance(
            movement_gdf=movement,
            common_points=common_points,
            contours_gdf=contours[model],
            yearly_ds=yearly_datasets[model],
            baseline_year=BASELINE_YEAR,
        )

        rates = calculate_regressions(movement.copy(), contours[model])
        rate_outputs[model] = rates

    comparison = gpd.GeoDataFrame(
        {
            "point_id": common_points["point_id"].to_numpy(),
            "alongshore_m": common_points["alongshore_m"].to_numpy(),
            "rate_eot20": rate_outputs["EOT20"]["rate_time"].to_numpy(),
            "rate_fes2022": rate_outputs["FES2022"]["rate_time"].to_numpy(),
            "geometry": common_points.geometry.to_numpy(),
        },
        crs=crs,
    )

    comparison["delta_rate"] = comparison["rate_fes2022"] - comparison["rate_eot20"]
    comparison["abs_delta_rate"] = comparison["delta_rate"].abs()
    comparison["sign_agree"] = (
        np.sign(comparison["rate_fes2022"]) == np.sign(comparison["rate_eot20"])
    )

    out_geojson = COMPARISON_ROOT / "rates" / "montrose_EOT20_vs_FES2022_rates_30m.geojson"
    comparison.to_crs("EPSG:4326").to_file(
        out_geojson,
        driver="GeoJSON",
        index=False,
    )

    valid = comparison.dropna(subset=["rate_eot20", "rate_fes2022"]).copy()

    if len(valid) < 2:
        raise RuntimeError("Not enough common valid rate points for comparison.")

    delta = valid["delta_rate"].to_numpy()
    eot = valid["rate_eot20"].to_numpy()
    fes = valid["rate_fes2022"].to_numpy()

    correlation = float(np.corrcoef(eot, fes)[0, 1])

    summary = {
        "n_common_points": int(len(comparison)),
        "n_valid_rate_pairs": int(len(valid)),
        "median_rate_eot20_m_per_yr": float(np.nanmedian(eot)),
        "median_rate_fes2022_m_per_yr": float(np.nanmedian(fes)),
        "median_delta_fes_minus_eot_m_per_yr": float(np.nanmedian(delta)),
        "median_absolute_delta_m_per_yr": float(np.nanmedian(np.abs(delta))),
        "mean_absolute_delta_m_per_yr": float(np.nanmean(np.abs(delta))),
        "rmse_rate_difference_m_per_yr": float(np.sqrt(np.nanmean(delta ** 2))),
        "delta_p10_m_per_yr": float(np.nanpercentile(delta, 10)),
        "delta_p90_m_per_yr": float(np.nanpercentile(delta, 90)),
        "rate_correlation_r": correlation,
        "sign_agreement_percent": float(valid["sign_agree"].mean() * 100.0),
        "sign_flip_percent": float((~valid["sign_agree"]).mean() * 100.0),
    }

    summary_file = COMPARISON_ROOT / "summaries" / "montrose_rate_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("RATE COMPARISON SUMMARY")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    make_rate_comparison_plots(comparison, valid, yearly_datasets["EOT20"])

    return comparison, summary


# =====================================================================
# RATE COMPARISON FIGURES
# =====================================================================


def make_rate_comparison_plots(comparison, valid, background_ds):
    # 1. EOT20 vs FES2022 rate scatter.
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(valid.rate_eot20, valid.rate_fes2022, s=20, alpha=0.7)

    limits = [
        float(min(valid.rate_eot20.min(), valid.rate_fes2022.min())),
        float(max(valid.rate_eot20.max(), valid.rate_fes2022.max())),
    ]
    ax.plot(limits, limits, "--", linewidth=1.0, color="0.35")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("EOT20 rate (m/year)")
    ax.set_ylabel("FES2022 rate (m/year)")
    ax.set_title("Montrose shoreline-change rate sensitivity")
    fig.tight_layout()
    fig.savefig(
        COMPARISON_ROOT / "qa" / "montrose_rate_EOT20_vs_FES2022_scatter.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 2. Difference along the common baseline.
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(comparison.alongshore_m, comparison.delta_rate, linewidth=1.1)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="0.35")
    ax.set_xlabel("Distance along common 2020 baseline (m)")
    ax.set_ylabel("FES2022 - EOT20 rate (m/year)")
    ax.set_title("Spatial sensitivity of shoreline-change rate to tide model")
    fig.tight_layout()
    fig.savefig(
        COMPARISON_ROOT / "qa" / "montrose_rate_difference_alongshore.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. Map of rate difference over the EOT20 baseline-year MNDWI composite.
    background = background_ds[WATER_INDEX].sel(year=BASELINE_YEAR)
    finite_delta = comparison.delta_rate.replace([np.inf, -np.inf], np.nan).dropna()

    if len(finite_delta) > 0:
        colour_limit = float(np.nanpercentile(np.abs(finite_delta), 95))
        if colour_limit == 0:
            colour_limit = 1.0

        norm = TwoSlopeNorm(vmin=-colour_limit, vcenter=0, vmax=colour_limit)

        fig, ax = plt.subplots(figsize=(7.5, 9.0))
        background.plot(
            ax=ax,
            cmap="Greys",
            vmin=-1,
            vmax=1,
            alpha=0.35,
            add_colorbar=False,
            zorder=1,
        )

        sc = ax.scatter(
            comparison.geometry.x,
            comparison.geometry.y,
            c=comparison.delta_rate,
            cmap="coolwarm_r",
            norm=norm,
            s=24,
            edgecolors="none",
            zorder=2,
        )

        ax.set_aspect("equal")
        ax.set_title("Rate difference: FES2022 - EOT20")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.025)
        cbar.set_label("Rate difference (m/year)")

        fig.tight_layout()
        fig.savefig(
            COMPARISON_ROOT / "qa" / "montrose_rate_difference_map.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)


# =====================================================================
# CUTOFF SUMMARY
# =====================================================================


def cutoff_summary(configs):
    rows = []
    for model, config in configs.items():
        with xr.open_dataset(config["cutoff_file"]) as ds:
            lo = ds["tide_cutoff_min"]
            hi = ds["tide_cutoff_max"]
            rows.append(
                {
                    "model": model,
                    "mean_lower_cutoff_m": float(lo.mean().values),
                    "mean_upper_cutoff_m": float(hi.mean().values),
                    "min_lower_cutoff_m": float(lo.min().values),
                    "max_upper_cutoff_m": float(hi.max().values),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(
        COMPARISON_ROOT / "summaries" / "montrose_tidal_cutoff_comparison.csv",
        index=False,
    )
    return df


# =====================================================================
# MAIN
# =====================================================================


def main():
    args = parse_args()
    ensure_dirs()
    configs = experiment_configs(args)

    heading("MONTROSE DEA COASTLINES: EOT20 vs FES2022")
    print(f"Tide-model directory: {args.tide_model_dir}")
    print(f"EOT20 raster version:   {configs['EOT20']['raster_version']}")
    print(f"FES2022 raster version: {configs['FES2022']['raster_version']}")

    if not args.skip_raster_generation:
        for model in ["EOT20", "FES2022"]:
            build_model_rasters(configs[model])
    else:
        print("Skipping raster generation by request.")

    # Ensure cutoffs exist even if raster generation was skipped.
    for model in ["EOT20", "FES2022"]:
        if not Path(configs[model]["cutoff_file"]).exists():
            raise FileNotFoundError(
                f"Missing cutoff file for {model}: {configs[model]['cutoff_file']}"
            )

    print()
    print("TIDAL CUTOFF COMPARISON")
    print(cutoff_summary(configs).to_string(index=False))

    # Always regenerate shorelines: this stage is cheap and prevents stale
    # vector products after a raster change.
    shoreline_files = {}
    yearly_datasets = {}

    for model in ["EOT20", "FES2022"]:
        shoreline_file, yearly_ds = extract_model_shorelines(configs[model])
        shoreline_files[model] = shoreline_file
        yearly_datasets[model] = yearly_ds

    obs_df, scene_mismatch = observation_comparison(configs, yearly_datasets)

    mismatch_years = obs_df.loc[
        ~obs_df["source_scene_sets_identical"], "year"
    ].tolist()

    if scene_mismatch:
        message = (
            "EOT20 and FES2022 source Landsat timestamp sets differ in years: "
            f"{mismatch_years}. A tide-model sensitivity experiment should use "
            "identical source scenes."
        )
        if args.allow_scene_mismatch:
            print("WARNING: " + message)
        else:
            raise RuntimeError(
                message
                + " Re-run with --fresh-eot20 to rebuild EOT20 against the current "
                "STAC catalogue, or use --allow-scene-mismatch only if you have "
                "verified the difference is harmless."
            )

    calculate_common_point_rates(
        configs=configs,
        shoreline_files=shoreline_files,
        yearly_datasets=yearly_datasets,
    )

    manifest = {
        "study_area": STUDY_AREA,
        "bbox": BBOX,
        "analysis_years": [START_YEAR, END_YEAR],
        "cutoff_years": [CUTOFF_START_YEAR, CUTOFF_END_YEAR],
        "baseline_year": BASELINE_YEAR,
        "point_spacing_m": POINT_SPACING_M,
        "water_index": WATER_INDEX,
        "index_threshold": INDEX_THRESHOLD,
        "tide_model_dir": args.tide_model_dir,
        "models": {
            model: {
                "raster_version": str(config["raster_version"]),
                "cache_root": str(config["cache_root"]),
                "cutoff_file": str(config["cutoff_file"]),
                "shoreline_file": str(shoreline_files[model]),
            }
            for model, config in configs.items()
        },
    }

    with open(
        COMPARISON_ROOT / "experiment_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    heading("TIDE-MODEL COMPARISON COMPLETE")
    print(f"Outputs: {COMPARISON_ROOT}")
    print()
    print("Key files:")
    print("  summaries/montrose_rate_comparison_summary.json")
    print("  summaries/montrose_observation_comparison.csv")
    print("  rates/montrose_EOT20_vs_FES2022_rates_30m.geojson")
    print("  qa/montrose_rate_EOT20_vs_FES2022_scatter.png")
    print("  qa/montrose_rate_difference_alongshore.png")
    print("  qa/montrose_rate_difference_map.png")


if __name__ == "__main__":
    main()
