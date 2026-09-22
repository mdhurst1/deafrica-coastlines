from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import planetary_computer.sas as pc_sas
import rasterio
import xarray as xr

from PIL import Image
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

start_year = 1988
end_year = 2025

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


# ---------------------------------------------------------------------
# Output folders
# ---------------------------------------------------------------------

output_dir = Path(
    "outputs/montrose_animation"
)

composite_dir = (
    output_dir / "composites"
)

frame_dir = (
    output_dir / "frames"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

composite_dir.mkdir(
    parents=True,
    exist_ok=True,
)

frame_dir.mkdir(
    parents=True,
    exist_ok=True,
)


gif_file = (
    output_dir
    / "montrose_mndwi_1988_2025.gif"
)

frame_duration_ms = 500


# ---------------------------------------------------------------------
# Remote-read retry settings
# ---------------------------------------------------------------------

max_attempts = 3

retry_wait_seconds = 5


# =====================================================================
# LOAD FIXED LONG-TERM TIDAL CUTOFFS
# =====================================================================

if not cutoff_file.exists():

    raise FileNotFoundError(
        f"Could not find {cutoff_file}. "
        "Calculate the long-term tidal cutoffs first."
    )


print(
    f"Loading fixed tidal cutoffs "
    f"from {cutoff_file}..."
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


# =====================================================================
# HELPER FUNCTION
# =====================================================================

def load_and_filter_year(
    year,
    tides,
    cutoff_min,
    cutoff_max,
):
    """
    Reload one year's Landsat imagery immediately before computation.

    This ensures Planetary Computer URLs are freshly signed.

    The first attempts fail loudly. On the final attempt, individual
    unreadable remote assets are treated as nodata rather than causing
    the entire multi-year animation job to fail.
    """

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        print()
        print(
            f"Raster download attempt "
            f"{attempt}/{max_attempts}"
        )


        # -------------------------------------------------------------
        # Force Planetary Computer to obtain a fresh SAS token
        # -------------------------------------------------------------

        pc_sas.TOKEN_CACHE.clear()


        # -------------------------------------------------------------
        # On the final attempt, tolerate an individual failed COG
        # -------------------------------------------------------------

        fail_on_error = (
            attempt < max_attempts
        )

        if fail_on_error:

            print(
                "Remote read mode: strict"
            )

        else:

            print(
                "Remote read mode: tolerate "
                "individual unreadable assets"
            )


        # -------------------------------------------------------------
        # Build a completely fresh Landsat load
        # -------------------------------------------------------------

        print(
            "Reloading Landsat imagery "
            "with fresh credentials..."
        )

        ds = load_water_index_stac(
            bbox=bbox,
            datetime=(
                f"{year}-01-01/"
                f"{year}-12-31"
            ),
            platforms=platforms,
            fail_on_error=fail_on_error,
        )


        # -------------------------------------------------------------
        # Make sure tide timestamps exactly match refreshed imagery
        # -------------------------------------------------------------

        tides_year = tides.sel(
            time=ds.time
        )

        ds["tide_m"] = tides_year


        # -------------------------------------------------------------
        # Immediately trigger the raster reads
        # -------------------------------------------------------------

        try:

            print(
                "Applying long-term tidal filter..."
            )

            tidal_ds = load_tidal_subset(
                year_ds=ds,
                tide_cutoff_min=cutoff_min,
                tide_cutoff_max=cutoff_max,
            )

            return tidal_ds


        except (
            rasterio.errors.RasterioIOError,
            rasterio.errors.RasterBlockError,
            rasterio.errors.WarpOperationError,
        ) as error:

            print()
            print(
                f"Remote raster read failed "
                f"for {year}:"
            )

            print(error)

            if attempt == max_attempts:

                raise

            print(
                f"Waiting {retry_wait_seconds} "
                "seconds before retrying..."
            )

            time.sleep(
                retry_wait_seconds
            )


# =====================================================================
# PROCESS EACH YEAR
# =====================================================================

frame_files = []


for year in range(
    start_year,
    end_year + 1,
):

    print()
    print("=" * 70)
    print(f"PROCESSING {year}")
    print("=" * 70)


    composite_file = (
        composite_dir
        / f"montrose_mndwi_composite_{year}.nc"
    )

    frame_file = (
        frame_dir
        / f"montrose_mndwi_{year}.png"
    )


    # =================================================================
    # USE CACHED COMPOSITE WHEN AVAILABLE
    # =================================================================

    if composite_file.exists():

        print(
            f"Using cached composite: "
            f"{composite_file}"
        )

        annual = xr.open_dataset(
            composite_file
        )


    else:

        # =============================================================
        # STEP 1
        # Build lightweight Landsat dataset for timestamps / geometry
        # =============================================================

        print(
            f"Finding Landsat observations "
            f"for {year}..."
        )

        try:

            ds_meta = load_water_index_stac(
                bbox=bbox,
                datetime=(
                    f"{year}-01-01/"
                    f"{year}-12-31"
                ),
                platforms=platforms,
            )

        except RuntimeError as error:

            print(
                f"Skipping {year}: {error}"
            )

            continue


        n_observations = (
            ds_meta.sizes["time"]
        )

        print(
            f"Found {n_observations} "
            "Landsat observations."
        )


        # =============================================================
        # STEP 2
        # Model tides BEFORE downloading imagery
        # =============================================================

        print(
            "Modelling pixel tides..."
        )

        tides = pixel_tides(
            data=ds_meta,
            model="EOT20",
            directory=tide_model_dir,
            resample=True,
        )


        # =============================================================
        # STEP 3
        # Align fixed 1984–2025 cutoff surfaces
        # =============================================================

        cutoff_min = (
            tide_cutoff_min
            .interp_like(
                tides.isel(time=0),
                method="linear",
            )
        )

        cutoff_max = (
            tide_cutoff_max
            .interp_like(
                tides.isel(time=0),
                method="linear",
            )
        )


        # =============================================================
        # STEP 4
        # Reload Landsat with fresh signing and actually read pixels
        # =============================================================

        tidal_ds = load_and_filter_year(
            year=year,
            tides=tides,
            cutoff_min=cutoff_min,
            cutoff_max=cutoff_max,
        )


        if tidal_ds.sizes["time"] == 0:

            print(
                f"No observations survive "
                f"tidal filtering in {year}."
            )

            continue


        print(
            f"{tidal_ds.sizes['time']} "
            "observations remain after "
            "tidal filtering."
        )


        # =============================================================
        # STEP 5
        # Generate annual composite
        # =============================================================

        print(
            "Generating annual composite..."
        )

        composite = tidal_composite(
            year_ds=tidal_ds,
            label=year,
            label_dim="year",
            output_dir=".",
            export_geotiff=False,
        )


        annual = composite.sel(
            year=year
        )


        # -------------------------------------------------------------
        # Everything should now be local/in memory
        # -------------------------------------------------------------

        annual = annual.compute()


        # =============================================================
        # STEP 6
        # Save composite to disk
        # =============================================================

        annual.to_netcdf(
            composite_file
        )

        print(
            f"Saved composite: "
            f"{composite_file}"
        )


    # =================================================================
    # CREATE ANIMATION FRAME
    # =================================================================

    print(
        f"Creating animation frame "
        f"for {year}..."
    )


    mndwi = annual[
        "mndwi"
    ].load()


    # -----------------------------------------------------------------
    # QA number shown on frame
    # -----------------------------------------------------------------

    if "count" in annual:

        mean_count = float(
            annual["count"]
            .mean(
                skipna=True
            )
            .values
        )

    else:

        mean_count = np.nan


    # -----------------------------------------------------------------
    # Plot annual MNDWI
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    mndwi.plot(
        ax=ax,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        add_colorbar=True,
        cbar_kwargs={
            "label": "MNDWI",
        },
    )


    ax.set_title(
        f"Montrose Bay annual Landsat composite — {year}\n"
        "Tidally filtered median MNDWI",
        fontsize=14,
    )


    # -----------------------------------------------------------------
    # Draw MNDWI = 0 candidate shoreline
    # -----------------------------------------------------------------

    try:

        ax.contour(
            mndwi.x,
            mndwi.y,
            mndwi.values,
            levels=[0],
            linewidths=1,
        )

    except Exception as error:

        print(
            f"Could not draw zero contour "
            f"for {year}: {error}"
        )


    # -----------------------------------------------------------------
    # QA annotation
    # -----------------------------------------------------------------

    if np.isfinite(
        mean_count
    ):

        ax.text(
            0.02,
            0.02,
            (
                "Mean valid observations: "
                f"{mean_count:.1f}"
            ),
            transform=ax.transAxes,
            fontsize=10,
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "none",
            },
        )


    ax.set_xlabel(
        "Easting (m)"
    )

    ax.set_ylabel(
        "Northing (m)"
    )


    plt.tight_layout()


    plt.savefig(
        frame_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


    print(
        f"Saved frame: "
        f"{frame_file}"
    )


    frame_files.append(
        frame_file
    )


    # -----------------------------------------------------------------
    # Close cached xarray file handles
    # -----------------------------------------------------------------

    if hasattr(
        annual,
        "close",
    ):

        annual.close()


# =====================================================================
# CREATE GIF
# =====================================================================

if len(frame_files) == 0:

    raise RuntimeError(
        "No animation frames were created."
    )


print()
print("=" * 70)
print("CREATING GIF")
print("=" * 70)


frame_files = sorted(
    frame_files
)


images = [
    Image.open(
        frame
    ).convert("RGB")
    for frame in frame_files
]


images[0].save(
    gif_file,
    save_all=True,
    append_images=images[1:],
    duration=frame_duration_ms,
    loop=0,
)


for image in images:

    image.close()


print()
print(
    f"Saved animation: "
    f"{gif_file}"
)

print(
    f"Frames included: "
    f"{len(frame_files)}"
)

print(
    f"Years requested: "
    f"{start_year}–{end_year}"
)

print()
print("Done.")