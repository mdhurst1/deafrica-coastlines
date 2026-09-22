from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from eo_tides.eo import pixel_tides

from coastlines.stac import load_water_index_stac


# =====================================================================
# SETTINGS
# =====================================================================

bbox = [-2.52, 56.70, -2.42, 56.76]

start_year = 1984
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


# =====================================================================
# OUTPUT / CACHE
# =====================================================================

output_dir = Path(
    "outputs/montrose_tide_QA"
)

cache_dir = (
    output_dir / "cache"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

cache_dir.mkdir(
    parents=True,
    exist_ok=True,
)


csv_file = (
    output_dir
    / "montrose_annual_tide_QA_1984_2025.csv"
)

plot_file = (
    output_dir
    / "montrose_median_contributing_tide_1984_2025.png"
)


# =====================================================================
# LOAD FIXED LONG-TERM TIDAL CUTOFFS
# =====================================================================

if not cutoff_file.exists():

    raise FileNotFoundError(
        f"Could not find {cutoff_file}. "
        "Run the long-term cutoff calculation first."
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
    "Loaded long-term tidal cutoffs."
)

print(
    f"Mean cutoff window: "
    f"{float(tide_cutoff_min.mean()):.2f} to "
    f"{float(tide_cutoff_max.mean()):.2f} m"
)


# =====================================================================
# HELPER: SUMMARISE CACHED YEAR
# =====================================================================

def summarise_cached_year(
    cache_file,
    year,
):

    with xr.open_dataset(
        cache_file
    ) as cached:

        # -------------------------------------------------------------
        # Require the new cache format
        # -------------------------------------------------------------

        if "accepted_image" not in cached:

            raise ValueError(
                "OLD_CACHE"
            )


        annual_tide = cached[
            "median_tide_m"
        ].load()

        accepted_count_map = cached[
            "accepted_count"
        ].load()

        accepted_image = cached[
            "accepted_image"
        ].load()


        n_scenes = int(
            cached.attrs[
                "n_scenes"
            ]
        )

        accepted_fraction = float(
            cached.attrs[
                "accepted_fraction"
            ]
        )


    # -----------------------------------------------------------------
    # Number of accepted Landsat images
    # -----------------------------------------------------------------

    n_accepted_images = int(
        accepted_image
        .sum()
        .values
    )


    # -----------------------------------------------------------------
    # Mean number of accepted observations per pixel
    # -----------------------------------------------------------------

    mean_accepted_per_pixel = float(
        accepted_count_map
        .mean(
            skipna=True
        )
        .values
    )


    # -----------------------------------------------------------------
    # Spatial tide statistics
    # -----------------------------------------------------------------

    values = (
        annual_tide
        .values
        .ravel()
    )

    values = values[
        np.isfinite(values)
    ]


    if len(values) == 0:

        median_tide = np.nan
        p10_tide = np.nan
        p90_tide = np.nan

    else:

        median_tide = float(
            np.median(values)
        )

        p10_tide = float(
            np.percentile(
                values,
                10,
            )
        )

        p90_tide = float(
            np.percentile(
                values,
                90,
            )
        )


    return {
        "year": year,
        "n_scenes": n_scenes,
        "n_accepted_images": n_accepted_images,
        "mean_accepted_per_pixel": mean_accepted_per_pixel,
        "median_tide_m": median_tide,
        "p10_tide_m": p10_tide,
        "p90_tide_m": p90_tide,
        "accepted_fraction": accepted_fraction,
    }


# =====================================================================
# PROCESS YEARS
# =====================================================================

results = []


for year in range(
    start_year,
    end_year + 1,
):

    print()
    print("=" * 70)
    print(year)
    print("=" * 70)


    cache_file = (
        cache_dir
        / f"montrose_tide_QA_{year}.nc"
    )


    # =================================================================
    # USE EXISTING CACHE
    # =================================================================

    if cache_file.exists():

        try:

            result = summarise_cached_year(
                cache_file=cache_file,
                year=year,
            )

            print(
                f"Using cached tide QA: "
                f"{cache_file}"
            )

            print(
                f"Accepted images: "
                f"{result['n_accepted_images']} / "
                f"{result['n_scenes']}"
            )

            print(
                f"Mean accepted observations "
                f"per pixel: "
                f"{result['mean_accepted_per_pixel']:.1f}"
            )

            print(
                f"Median contributing tide: "
                f"{result['median_tide_m']:.3f} m"
            )

            print(
                f"Spatial 10–90% range: "
                f"{result['p10_tide_m']:.3f} to "
                f"{result['p90_tide_m']:.3f} m"
            )


            results.append(
                result
            )


            # ---------------------------------------------------------
            # Save progress immediately
            # ---------------------------------------------------------

            pd.DataFrame(
                results
            ).to_csv(
                csv_file,
                index=False,
            )


            continue


        except ValueError as error:

            if str(error) == "OLD_CACHE":

                print(
                    "Existing cache is from the old format "
                    "and does not contain per-image acceptance."
                )

                print(
                    f"Recomputing {year}..."
                )

            else:

                raise


    # =================================================================
    # FIND LANDSAT OBSERVATIONS
    # =================================================================

    try:

        ds = load_water_index_stac(
            bbox=bbox,
            datetime=(
                f"{year}-01-01/"
                f"{year}-12-31"
            ),
            platforms=platforms,
        )


    except RuntimeError:

        print(
            f"No Landsat observations "
            f"for {year}."
        )


        results.append(
            {
                "year": year,
                "n_scenes": 0,
                "n_accepted_images": 0,
                "mean_accepted_per_pixel": np.nan,
                "median_tide_m": np.nan,
                "p10_tide_m": np.nan,
                "p90_tide_m": np.nan,
                "accepted_fraction": np.nan,
            }
        )


        pd.DataFrame(
            results
        ).to_csv(
            csv_file,
            index=False,
        )


        continue


    n_scenes = int(
        ds.sizes["time"]
    )


    print(
        f"Found {n_scenes} "
        "Landsat images."
    )


    # =================================================================
    # MODEL TIDES
    # =================================================================

    print(
        "Modelling pixel tides..."
    )


    tides = pixel_tides(
        data=ds,
        model="EOT20",
        directory=tide_model_dir,
        resample=True,
    )


    # =================================================================
    # ALIGN FIXED 1984–2025 CUT-OFF SURFACES
    # =================================================================

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


    # =================================================================
    # CREATE TIDAL ACCEPTANCE MASK
    # =================================================================

    accepted = (
        (tides >= cutoff_min)
        &
        (tides <= cutoff_max)
    )


    # =================================================================
    # DETERMINE WHICH LANDSAT IMAGES ARE ACCEPTED
    #
    # This reproduces the logic used by load_tidal_subset():
    #
    #     tide_bool.sum(dim=["x", "y"]) > 0
    #
    # An image counts as accepted if at least one pixel in the AOI
    # lies inside the tidal acceptance window.
    # =================================================================

    accepted_image = (
        accepted
        .sum(
            dim=[
                "x",
                "y",
            ]
        )
        > 0
    )


    accepted_image = (
        accepted_image
        .compute()
    )


    n_accepted_images = int(
        accepted_image
        .sum()
        .values
    )


    print(
        f"Accepted Landsat images: "
        f"{n_accepted_images} / "
        f"{n_scenes}"
    )


    # =================================================================
    # PIXEL-LEVEL ACCEPTANCE QA
    # =================================================================

    accepted_fraction = float(
        accepted
        .mean()
        .values
    )


    accepted_count_map = (
        accepted
        .sum(
            dim="time"
        )
        .compute()
    )


    mean_accepted_per_pixel = float(
        accepted_count_map
        .mean(
            skipna=True
        )
        .values
    )


    print(
        f"Accepted tide pixels: "
        f"{accepted_fraction:.1%}"
    )

    print(
        f"Mean accepted observations "
        f"per pixel: "
        f"{mean_accepted_per_pixel:.1f}"
    )


    # =================================================================
    # ANNUAL MEDIAN CONTRIBUTING TIDE
    # =================================================================

    accepted_tides = tides.where(
        accepted
    )


    annual_tide = (
        accepted_tides
        .median(
            dim="time",
            skipna=True,
        )
        .compute()
    )


    # =================================================================
    # CACHE SPATIAL QA OUTPUT
    # =================================================================

    cache_ds = xr.Dataset(
        {
            "median_tide_m": annual_tide,
            "accepted_count": accepted_count_map,
            "accepted_image": accepted_image,
        }
    )


    cache_ds[
        "median_tide_m"
    ].attrs = {
        "long_name": (
            "Annual median contributing "
            "tide height"
        ),
        "units": "m",
        "reference": (
            "EOT20 mean sea level"
        ),
    }


    cache_ds[
        "accepted_count"
    ].attrs = {
        "long_name": (
            "Number of tide-accepted "
            "Landsat observations at each pixel"
        ),
        "units": "observations",
    }


    cache_ds[
        "accepted_image"
    ].attrs = {
        "long_name": (
            "Whether Landsat image contains "
            "at least one tide-accepted pixel"
        ),
    }


    cache_ds.attrs = {
        "year": year,
        "n_scenes": n_scenes,
        "n_accepted_images": (
            n_accepted_images
        ),
        "accepted_fraction": (
            accepted_fraction
        ),
        "tide_model": "EOT20",
        "tide_cutoff_period": (
            "1984-2025"
        ),
    }


    cache_ds.to_netcdf(
        cache_file
    )


    print(
        f"Cached tide QA: "
        f"{cache_file}"
    )


    # =================================================================
    # SPATIAL TIDE STATISTICS
    # =================================================================

    values = (
        annual_tide
        .values
        .ravel()
    )

    values = values[
        np.isfinite(values)
    ]


    if len(values) == 0:

        median_tide = np.nan
        p10_tide = np.nan
        p90_tide = np.nan

    else:

        median_tide = float(
            np.median(values)
        )

        p10_tide = float(
            np.percentile(
                values,
                10,
            )
        )

        p90_tide = float(
            np.percentile(
                values,
                90,
            )
        )


    print(
        f"Median contributing tide: "
        f"{median_tide:.3f} m"
    )

    print(
        f"Spatial 10–90% range: "
        f"{p10_tide:.3f} to "
        f"{p90_tide:.3f} m"
    )


    # =================================================================
    # STORE YEAR RESULTS
    # =================================================================

    results.append(
        {
            "year": year,
            "n_scenes": n_scenes,
            "n_accepted_images": (
                n_accepted_images
            ),
            "mean_accepted_per_pixel": (
                mean_accepted_per_pixel
            ),
            "median_tide_m": (
                median_tide
            ),
            "p10_tide_m": (
                p10_tide
            ),
            "p90_tide_m": (
                p90_tide
            ),
            "accepted_fraction": (
                accepted_fraction
            ),
        }
    )


    # =================================================================
    # SAVE CSV AFTER EVERY YEAR
    # =================================================================

    pd.DataFrame(
        results
    ).to_csv(
        csv_file,
        index=False,
    )


    print(
        f"Updated QA table: "
        f"{csv_file}"
    )


# =====================================================================
# FINAL DATAFRAME
# =====================================================================

df = (
    pd.DataFrame(
        results
    )
    .sort_values(
        "year"
    )
)


df.to_csv(
    csv_file,
    index=False,
)


print()
print("=" * 70)
print("ANNUAL TIDE QA")
print("=" * 70)

print(
    df.to_string(
        index=False
    )
)


# =====================================================================
# CREATE QA PLOT
# =====================================================================

valid = df.dropna(
    subset=[
        "median_tide_m",
        "n_accepted_images",
        "p10_tide_m",
        "p90_tide_m",
    ]
)


fig, ax = plt.subplots(
    figsize=(13, 6)
)


# =====================================================================
# COLOUR USED FOR LINE, DOTS, SHADING AND SIZE LEGEND
# =====================================================================

data_colour = "C0"


# =====================================================================
# CONNECTING LINE
# =====================================================================

ax.plot(
    valid["year"],
    valid["median_tide_m"],
    linewidth=1.2,
    color=data_colour,
    label="Median contributing tide",
)


# =====================================================================
# SPATIAL 10–90% RANGE
# =====================================================================

ax.fill_between(
    valid["year"],
    valid["p10_tide_m"],
    valid["p90_tide_m"],
    color=data_colour,
    alpha=0.20,
    label="Spatial 10–90% range",
)


# =====================================================================
# MEAN SEA LEVEL
# =====================================================================

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    label="Mean sea level",
)


# =====================================================================
# DATA POINTS
#
# Marker AREA is proportional to number of accepted Landsat images.
#
# A year with 20 accepted images therefore has twice the circle area
# of a year with 10 accepted images.
# =====================================================================

size_scale = 18


marker_sizes = (
    valid[
        "n_accepted_images"
    ]
    * size_scale
)


ax.scatter(
    valid["year"],
    valid["median_tide_m"],
    s=marker_sizes,
    color=data_colour,
    zorder=4,
)


# =====================================================================
# MAIN LEGEND
# =====================================================================

main_legend = ax.legend(
    loc="upper right",
)

ax.add_artist(
    main_legend
)


# =====================================================================
# CIRCLE-SIZE LEGEND
# =====================================================================

min_count = int(
    valid[
        "n_accepted_images"
    ].min()
)

median_count = int(
    np.median(
        valid[
            "n_accepted_images"
        ]
    )
)

max_count = int(
    valid[
        "n_accepted_images"
    ].max()
)


count_examples = sorted(
    set(
        [
            min_count,
            median_count,
            max_count,
        ]
    )
)


size_handles = []


for count in count_examples:

    handle = ax.scatter(
        [],
        [],
        s=(
            count
            * size_scale
        ),
        color=data_colour,
        label=(
            f"{count} images"
        ),
    )

    size_handles.append(
        handle
    )


size_legend = ax.legend(
    handles=size_handles,
    title="Accepted Landsat images per year",
    loc="lower right",
    ncol=len(size_handles),
    columnspacing=2.5,
    handletextpad=1.0,
    borderpad=1.0,
    scatterpoints=1,
)


ax.add_artist(
    size_legend
)


# =====================================================================
# FORMATTING
# =====================================================================

ax.set_xlabel(
    "Year"
)

ax.set_ylabel(
    "Median contributing tide height "
    "relative to mean sea level (m)"
)


ax.set_title(
    "Residual tidal sampling bias in annual Landsat composites\n"
    "Circle area proportional to number of accepted Landsat images"
)


ax.set_xlim(
    start_year - 0.5,
    end_year + 0.5,
)


ax.grid(
    axis="y",
    alpha=0.3,
)


plt.tight_layout()


# =====================================================================
# SAVE FIGURE
# =====================================================================

plt.savefig(
    plot_file,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print()
print(
    f"Saved QA table: "
    f"{csv_file}"
)

print(
    f"Saved plot: "
    f"{plot_file}"
)

print()
print("Done.")