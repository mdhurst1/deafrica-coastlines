from pathlib import Path
import inspect
import time

import numpy as np
import planetary_computer.sas as pc_sas
import rasterio
import xarray as xr

from eo_tides.eo import pixel_tides

from coastlines.stac import load_water_index_stac
from coastlines.raster import (
    load_tidal_subset,
    tidal_composite,
)


# =====================================================================
# SETTINGS
# =====================================================================

bbox = [-2.52, 56.70, -2.42, 56.76]

# Final annual products required
start_year = 1988
end_year = 2025

# DEA three-year gapfill requires:
#
# previous year + current year + following year
#
# Therefore 1988 needs 1987 data,
# and 2025 needs 2026 data.
context_start_year = start_year - 1
context_end_year = end_year + 1


tide_model_dir = "/home/mh322u/tide_models"

cutoff_file = Path(
    "montrose_tidal_cutoffs_1984_2025.nc"
)

platforms = [
    "landsat-5",
    "landsat-7",
    "landsat-8",
    "landsat-9",
]


# =====================================================================
# DEA-STYLE OUTPUT STRUCTURE
# =====================================================================

raster_root = Path(
    "outputs/dea_rasters"
)

raster_version = "scotland_v0.1"

study_area = "montrose"


output_dir = (
    raster_root
    / raster_version
    / f"{study_area}_{raster_version}"
)


output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# CACHE STRUCTURE
# =====================================================================

cache_root = Path(
    "outputs/montrose_dea_cache"
)

tide_cache_dir = (
    cache_root / "tides"
)

observation_cache_dir = (
    cache_root / "tidal_observations"
)


tide_cache_dir.mkdir(
    parents=True,
    exist_ok=True,
)

observation_cache_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# RETRY SETTINGS FOR PLANETARY COMPUTER
# =====================================================================

max_attempts = 3

retry_wait_seconds = 5


# =====================================================================
# LOAD FIXED 1984–2025 TIDAL CUTOFFS
# =====================================================================

if not cutoff_file.exists():

    raise FileNotFoundError(
        f"Could not find {cutoff_file}"
    )


print(
    f"Loading fixed tidal cutoffs from "
    f"{cutoff_file}"
)


cutoffs = xr.open_dataset(
    cutoff_file
)


tide_cutoff_min = cutoffs[
    "tide_cutoff_min"
]

tide_cutoff_max = cutoffs[
    "tide_cutoff_max"
]


print(
    f"Mean tidal window: "
    f"{float(tide_cutoff_min.mean()):.2f} to "
    f"{float(tide_cutoff_max.mean()):.2f} m"
)


# =====================================================================
# CHECK WHETHER STAC LOADER SUPPORTS fail_on_error
# =====================================================================

loader_signature = inspect.signature(
    load_water_index_stac
)

supports_fail_on_error = (
    "fail_on_error"
    in loader_signature.parameters
)


print(
    f"STAC loader supports fail_on_error: "
    f"{supports_fail_on_error}"
)


# =====================================================================
# HELPERS
# =====================================================================

def stac_load_year(
    year,
    fail_on_error=True,
):
    """
    Create a lazy Landsat dataset for one year.
    """

    kwargs = {
        "bbox": bbox,
        "datetime": (
            f"{year}-01-01/"
            f"{year}-12-31"
        ),
        "platforms": platforms,
    }


    if supports_fail_on_error:

        kwargs[
            "fail_on_error"
        ] = fail_on_error


    return load_water_index_stac(
        **kwargs
    )


# ---------------------------------------------------------------------


def save_tide_cache(
    tides,
    tide_file,
):
    """
    Save modelled tide surfaces to disk.
    """

    tides = tides.rename(
        "tide_m"
    )


    tide_ds = tides.to_dataset()


    tide_ds.to_netcdf(
        tide_file
    )


    print(
        f"Saved tide cache: "
        f"{tide_file}"
    )


# ---------------------------------------------------------------------


def load_tide_cache(
    tide_file,
):
    """
    Load cached modelled tides fully into memory.
    """

    with xr.open_dataset(
        tide_file
    ) as ds:

        tides = ds[
            "tide_m"
        ].load()


    return tides


# ---------------------------------------------------------------------


def build_tides_for_year(
    year,
):
    """
    Model EOT20 tide height at every Landsat acquisition time.

    Landsat imagery itself is not downloaded here.
    """

    tide_file = (
        tide_cache_dir
        / f"{year}_tides.nc"
    )


    # -------------------------------------------------------------
    # Re-use tide model cache
    # -------------------------------------------------------------

    if tide_file.exists():

        print(
            f"Using cached tides: "
            f"{tide_file}"
        )

        return load_tide_cache(
            tide_file
        )


    # -------------------------------------------------------------
    # Query Landsat metadata/grid
    # -------------------------------------------------------------

    print(
        f"Querying Landsat observations "
        f"for tide modelling: {year}"
    )


    ds_meta = stac_load_year(
        year
    )


    n_scenes = int(
        ds_meta.sizes["time"]
    )


    print(
        f"{year}: "
        f"{n_scenes} Landsat observations"
    )


    # -------------------------------------------------------------
    # Model tides
    # -------------------------------------------------------------

    print(
        f"{year}: modelling EOT20 tides..."
    )


    tides = pixel_tides(
        data=ds_meta,
        model="EOT20",
        directory=tide_model_dir,
        resample=True,
    )


    tides = tides.rename(
        "tide_m"
    )


    # Tide modelling uses image geometry/time only,
    # not the Landsat pixel values.
    tides = tides.compute()


    save_tide_cache(
        tides=tides,
        tide_file=tide_file,
    )


    return tides


# ---------------------------------------------------------------------


def tide_times_match(
    tides,
    ds,
):
    """
    Test whether cached tides cover all Landsat acquisition times.
    """

    tide_times = set(
        tides.time.values.astype(
            "datetime64[ns]"
        )
    )

    image_times = set(
        ds.time.values.astype(
            "datetime64[ns]"
        )
    )


    return image_times.issubset(
        tide_times
    )


# ---------------------------------------------------------------------


def regenerate_tides(
    year,
    ds,
):
    """
    Regenerate tide cache if STAC contents have changed.
    """

    print(
        f"{year}: Landsat timestamps have changed; "
        f"regenerating tide cache."
    )


    tides = pixel_tides(
        data=ds,
        model="EOT20",
        directory=tide_model_dir,
        resample=True,
    )


    tides = tides.rename(
        "tide_m"
    ).compute()


    tide_file = (
        tide_cache_dir
        / f"{year}_tides.nc"
    )


    save_tide_cache(
        tides=tides,
        tide_file=tide_file,
    )


    return tides


# ---------------------------------------------------------------------


def build_filtered_observations(
    year,
):
    """
    Load Landsat NDWI and MNDWI for one year, attach modelled tides, apply the
    fixed 1984–2025 DEA tidal window, compute the pixels and cache the
    tide-filtered observations locally.

    Returns an in-memory Dataset containing:

        mndwi(time, y, x)
        tide_m(time, y, x)
    """

    observation_file = (
        observation_cache_dir
        / f"{year}_tidal.nc"
    )


    # -------------------------------------------------------------
    # Re-use fully downloaded, tide-filtered Landsat observations
    # -------------------------------------------------------------

    if observation_file.exists():

        print(
            f"Checking cached tide-filtered observations: "
            f"{observation_file}"
        )

        with xr.open_dataset(
            observation_file
        ) as cached:

            required_variables = {
                "mndwi",
                "ndwi",
                "tide_m",
            }

            available_variables = set(
                cached.data_vars
            )

            cache_valid = (
                required_variables
                .issubset(
                    available_variables
                )
            )

            if cache_valid:

                print(
                    f"Using cached tide-filtered observations: "
                    f"{observation_file}"
                )

                ds = cached.load()

                return ds


        # -------------------------------------------------------------
        # Old cache does not contain everything required
        # -------------------------------------------------------------

        print(
            f"{year}: cache is incomplete "
            f"(probably missing NDWI)."
        )

        print(
            f"{year}: deleting old cache and rebuilding..."
        )

        observation_file.unlink()


    # -------------------------------------------------------------
    # Obtain / model tide data
    # -------------------------------------------------------------

    tides = build_tides_for_year(
        year
    )


    # -------------------------------------------------------------
    # Try Landsat download several times
    # -------------------------------------------------------------

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        print()
        print(
            f"{year}: raster download attempt "
            f"{attempt}/{max_attempts}"
        )


        # ---------------------------------------------------------
        # Clear Planetary Computer's cached SAS token
        # ---------------------------------------------------------

        pc_sas.TOKEN_CACHE.clear()


        # ---------------------------------------------------------
        # On the final attempt, permit individual broken assets
        # if our STAC wrapper supports fail_on_error.
        # ---------------------------------------------------------

        strict = (
            attempt < max_attempts
        )


        print(
            f"{year}: remote read mode = "
            f"{'strict' if strict else 'tolerant'}"
        )


        try:

            # -----------------------------------------------------
            # Fresh STAC load immediately before reading pixels
            # -----------------------------------------------------

            ds = stac_load_year(
                year,
                fail_on_error=strict,
            )


            print(
                f"{year}: fresh Landsat stack contains "
                f"{ds.sizes['time']} observations"
            )


            # -----------------------------------------------------
            # If STAC has changed since tide cache was built,
            # regenerate tides
            # -----------------------------------------------------

            if not tide_times_match(
                tides,
                ds,
            ):

                tides = regenerate_tides(
                    year=year,
                    ds=ds,
                )


            # -----------------------------------------------------
            # Match tide timestamps exactly to Landsat
            # -----------------------------------------------------

            tides_year = tides.sel(
                time=ds.time
            )


            # -----------------------------------------------------
            # Only retain variables needed by DEA raster workflow
            # -----------------------------------------------------

            ds = ds[
                [
                    "mndwi",
                    "ndwi",
                ]
            ]


            ds[
                "tide_m"
            ] = tides_year


            # -----------------------------------------------------
            # Align fixed long-term cutoff grids
            # -----------------------------------------------------

            cutoff_min = (
                tide_cutoff_min
                .interp_like(
                    tides_year.isel(
                        time=0
                    ),
                    method="linear",
                )
            )


            cutoff_max = (
                tide_cutoff_max
                .interp_like(
                    tides_year.isel(
                        time=0
                    ),
                    method="linear",
                )
            )


            # -----------------------------------------------------
            # Apply DEA tidal filtering.
            #
            # load_tidal_subset() calls .compute(), so Landsat
            # imagery is actually downloaded here.
            # -----------------------------------------------------

            print(
                f"{year}: applying fixed "
                f"1984–2025 tidal filter..."
            )


            tidal_ds = load_tidal_subset(
                year_ds=ds,
                tide_cutoff_min=cutoff_min,
                tide_cutoff_max=cutoff_max,
            )


            print(
                f"{year}: "
                f"{tidal_ds.sizes['time']} acquisitions "
                f"contain accepted tidal pixels"
            )


            # -----------------------------------------------------
            # Save LOCAL observation cache
            # -----------------------------------------------------

            tidal_ds.to_netcdf(
                observation_file
            )


            print(
                f"{year}: saved tide-filtered "
                f"Landsat cache:"
            )

            print(
                f"    {observation_file}"
            )


            return tidal_ds


        except Exception as error:

            print()
            print(
                f"{year}: raster read failed:"
            )

            print(
                error
            )


            if attempt == max_attempts:

                raise


            print(
                f"{year}: waiting "
                f"{retry_wait_seconds} seconds "
                f"before retry..."
            )


            time.sleep(
                retry_wait_seconds
            )


    raise RuntimeError(
        f"Unable to process {year}"
    )


# ---------------------------------------------------------------------


def annual_files_exist(
    year,
):
    """
    Check whether all DEA annual raster outputs exist.
    """

    required = [
        output_dir
        / f"{year}_mndwi.tif",

        output_dir
        / f"{year}_ndwi.tif",

        output_dir
        / f"{year}_count.tif",

        output_dir
        / f"{year}_stdev.tif",

        output_dir
        / f"{year}_tide_m.tif",
    ]

    return all(
        file.exists()
        for file in required
    )

# ---------------------------------------------------------------------


def gapfill_files_exist(
    year,
):
    """
    Check whether all DEA three-year gapfill outputs exist.
    """

    required = [
        output_dir
        / f"{year}_mndwi_gapfill.tif",

        output_dir
        / f"{year}_ndwi_gapfill.tif",

        output_dir
        / f"{year}_count_gapfill.tif",

        output_dir
        / f"{year}_stdev_gapfill.tif",

        output_dir
        / f"{year}_tide_m_gapfill.tif",
    ]

    return all(
        file.exists()
        for file in required
    )


# =====================================================================
# STAGE 1
#
# DOWNLOAD + CACHE TIDE-FILTERED LANDSAT OBSERVATIONS
#
# We need one year either side of the requested annual period so that
# every requested year can have a true three-year DEA gapfill composite.
# =====================================================================

print()
print("=" * 75)
print("STAGE 1: CACHE TIDE-FILTERED LANDSAT OBSERVATIONS")
print("=" * 75)


for year in range(
    context_start_year,
    context_end_year + 1,
):

    print()
    print("-" * 75)
    print(
        f"CACHING {year}"
    )
    print("-" * 75)


    build_filtered_observations(
        year
    )


# =====================================================================
# STAGE 2
#
# CREATE ANNUAL + THREE-YEAR GAPFILL COMPOSITES
#
# This follows DEA Coastlines:
#
#   annual:
#       median(observations in year Y)
#
#   gapfill:
#       median(
#           observations in Y-1
#           + observations in Y
#           + observations in Y+1
#       )
#
# Importantly, this is NOT:
#
#       median(
#           annual median Y-1,
#           annual median Y,
#           annual median Y+1
#       )
#
# =====================================================================

print()
print("=" * 75)
print("STAGE 2: CREATE DEA ANNUAL + GAPFILL RASTERS")
print("=" * 75)


# ---------------------------------------------------------------------
# Load first three-year window
# ---------------------------------------------------------------------

previous_ds = build_filtered_observations(
    start_year - 1
)

current_ds = build_filtered_observations(
    start_year
)

future_ds = build_filtered_observations(
    start_year + 1
)


# ---------------------------------------------------------------------
# Progress through each target year
# ---------------------------------------------------------------------

for year in range(
    start_year,
    end_year + 1,
):

    print()
    print("=" * 75)
    print(
        f"COMPOSITING {year}"
    )
    print("=" * 75)


    # =================================================================
    # ANNUAL COMPOSITE
    # =================================================================

    if annual_files_exist(
        year
    ):

        print(
            f"{year}: annual DEA rasters already exist; "
            f"skipping."
        )


    else:

        print(
            f"{year}: generating annual composite..."
        )


        tidal_composite(
            year_ds=current_ds,
            label=year,
            label_dim="year",
            output_dir=str(
                output_dir
            ),
            export_geotiff=True,
        )


        print(
            f"{year}: annual rasters written."
        )


    # =================================================================
    # THREE-YEAR GAPFILL COMPOSITE
    # =================================================================

    if gapfill_files_exist(
        year
    ):

        print(
            f"{year}: gapfill DEA rasters already exist; "
            f"skipping."
        )


    else:

        print(
            f"{year}: combining "
            f"{year - 1}, {year}, {year + 1} "
            f"observations..."
        )


        gapfill_ds = xr.concat(
            [
                previous_ds,
                current_ds,
                future_ds,
            ],
            dim="time",
        )


        gapfill_ds = gapfill_ds.sortby(
            "time"
        )


        print(
            f"{year}: gapfill contains "
            f"{gapfill_ds.sizes['time']} "
            f"tide-filtered acquisitions."
        )


        tidal_composite(
            year_ds=gapfill_ds,
            label=year,
            label_dim="year",
            output_dir=str(
                output_dir
            ),
            output_suffix="_gapfill",
            export_geotiff=True,
        )


        print(
            f"{year}: gapfill rasters written."
        )


        del gapfill_ds


    # =================================================================
    # SHIFT THREE-YEAR WINDOW FOR NEXT YEAR
    # =================================================================

    if year < end_year:

        previous_ds = current_ds

        current_ds = future_ds


        next_year = (
            year + 2
        )


        future_ds = build_filtered_observations(
            next_year
        )


# =====================================================================
# FINISHED
# =====================================================================

print()
print("=" * 75)
print("DEA RASTER GENERATION COMPLETE")
print("=" * 75)

print()
print(
    f"Output directory:"
)

print(
    output_dir
)

print()

print(
    f"Annual products: "
    f"{start_year}–{end_year}"
)

print(
    "Each annual year contains:"
)

print(
    "    *_mndwi.tif"
)

print(
    "    *_count.tif"
)

print(
    "    *_stdev.tif"
)

print(
    "    *_tide_m.tif"
)

print()

print(
    "Each year also contains corresponding "
    "*_gapfill.tif products."
)

print()
print(
    "Finished."
)