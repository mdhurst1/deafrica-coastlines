from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


# Output folders
output_dir = Path("outputs/montrose_animation")
composite_dir = output_dir / "composites"
frame_dir = output_dir / "frames"

output_dir.mkdir(parents=True, exist_ok=True)
composite_dir.mkdir(parents=True, exist_ok=True)
frame_dir.mkdir(parents=True, exist_ok=True)


# Animation output
gif_file = output_dir / "montrose_mndwi_1988_2025.gif"


# Animation timing
frame_duration_ms = 500


# =====================================================================
# LOAD FIXED LONG-TERM TIDAL CUTOFFS
# =====================================================================

if not cutoff_file.exists():
    raise FileNotFoundError(
        f"Could not find {cutoff_file}. "
        "Calculate the long-term tidal cutoffs first."
    )

print(
    f"Loading fixed tidal cutoffs from {cutoff_file}..."
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
# PROCESS EACH YEAR
# =====================================================================

frame_files = []

for year in range(
    start_year,
    end_year + 1,
):

    print()
    print("=" * 70)
    print(f"Processing {year}")
    print("=" * 70)

    composite_file = (
        composite_dir
        / f"montrose_mndwi_composite_{year}.nc"
    )

    frame_file = (
        frame_dir
        / f"montrose_mndwi_{year}.png"
    )


    # -----------------------------------------------------------------
    # Load existing annual composite if already processed
    # -----------------------------------------------------------------

    if composite_file.exists():

        print(
            f"Using cached composite: "
            f"{composite_file}"
        )

        annual = xr.open_dataset(
            composite_file
        )

    else:

        # -------------------------------------------------------------
        # Load Landsat imagery for this year
        # -------------------------------------------------------------

        print(
            f"Loading Landsat data for {year}..."
        )

        try:

            ds = load_water_index_stac(
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


        print(
            f"Found {ds.sizes['time']} "
            f"Landsat observations."
        )


        # -------------------------------------------------------------
        # Model tide heights for each Landsat observation
        # -------------------------------------------------------------

        print(
            "Modelling pixel tides..."
        )

        tides = pixel_tides(
            data=ds,
            model="EOT20",
            directory=tide_model_dir,
            resample=True,
        )


        # -------------------------------------------------------------
        # Align fixed cutoffs to this year's grid
        # -------------------------------------------------------------

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


        # -------------------------------------------------------------
        # Attach tides using DEA Coastlines variable name
        # -------------------------------------------------------------

        ds["tide_m"] = tides


        # -------------------------------------------------------------
        # Apply fixed 1984–2025 tidal filter
        # -------------------------------------------------------------

        print(
            "Applying long-term tidal filter..."
        )

        tidal_ds = load_tidal_subset(
            year_ds=ds,
            tide_cutoff_min=cutoff_min,
            tide_cutoff_max=cutoff_max,
        )


        if tidal_ds.sizes["time"] == 0:

            print(
                f"No observations survive "
                f"tidal filtering in {year}."
            )

            continue


        print(
            f"{tidal_ds.sizes['time']} "
            f"observations remain after "
            f"tidal filtering."
        )


        # -------------------------------------------------------------
        # Generate annual DEA Coastlines composite
        # -------------------------------------------------------------

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
        # Load into memory before closing remote resources
        # -------------------------------------------------------------

        annual = annual.compute()


        # -------------------------------------------------------------
        # Save annual composite
        # -------------------------------------------------------------

        annual.to_netcdf(
            composite_file
        )

        print(
            f"Saved {composite_file}"
        )


    # =================================================================
    # CREATE ANIMATION FRAME
    # =================================================================

    print(
        f"Creating frame for {year}..."
    )


    # Ensure data loaded if coming from cached NetCDF
    mndwi = annual["mndwi"].load()


    # -------------------------------------------------------------
    # Useful QA numbers to put on frame
    # -------------------------------------------------------------

    if "count" in annual:

        mean_count = float(
            annual["count"]
            .mean(skipna=True)
            .values
        )

    else:

        mean_count = np.nan


    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    plot = mndwi.plot(
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
        f"Tidally filtered median MNDWI",
        fontsize=14,
    )


    # Small QA annotation
    if np.isfinite(mean_count):

        ax.text(
            0.02,
            0.02,
            (
                f"Mean valid observations: "
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
        f"Saved {frame_file}"
    )

    frame_files.append(
        frame_file
    )


# =====================================================================
# CREATE GIF
# =====================================================================

if len(frame_files) == 0:

    raise RuntimeError(
        "No animation frames were created."
    )


print()
print("=" * 70)
print("Creating animation")
print("=" * 70)


# Ensure chronological order
frame_files = sorted(
    frame_files
)


images = [
    Image.open(frame).convert("RGB")
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
    f"Saved animation: {gif_file}"
)

print(
    f"Frames included: {len(frame_files)}"
)

print(
    f"Years requested: "
    f"{start_year}–{end_year}"
)

print("\nDone.")