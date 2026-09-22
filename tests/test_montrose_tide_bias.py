from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from eo_tides.eo import pixel_tides

from coastlines.stac import load_water_index_stac


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Load fixed long-term tidal cutoffs
# ---------------------------------------------------------------------

if not cutoff_file.exists():
    raise FileNotFoundError(
        f"Could not find {cutoff_file}. "
        "Run the long-term cutoff calculation first."
    )

cutoffs = xr.open_dataset(cutoff_file)

tide_cutoff_min = cutoffs["tide_cutoff_min"]
tide_cutoff_max = cutoffs["tide_cutoff_max"]

print("Loaded long-term tidal cutoffs.")

print(
    f"Mean cutoff window: "
    f"{float(tide_cutoff_min.mean()):.2f} to "
    f"{float(tide_cutoff_max.mean()):.2f} m"
)


# ---------------------------------------------------------------------
# Loop through years
# ---------------------------------------------------------------------

results = []

for year in range(start_year, end_year + 1):

    print()
    print("=" * 60)
    print(year)
    print("=" * 60)

    try:

        # -------------------------------------------------------------
        # Load Landsat catalogue for this year
        #
        # This remains lazy: we mainly need timestamps and grid
        # geometry rather than the satellite imagery itself.
        # -------------------------------------------------------------

        ds = load_water_index_stac(
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            platforms=platforms,
        )

    except RuntimeError:

        print(f"No Landsat observations for {year}.")

        results.append(
            {
                "year": year,
                "n_scenes": 0,
                "accepted_count": np.nan,
                "median_tide_m": np.nan,
                "p10_tide_m": np.nan,
                "p90_tide_m": np.nan,
                "accepted_fraction": np.nan,
            }
        )

        continue


    n_scenes = ds.sizes["time"]

    print(
        f"Found {n_scenes} Landsat observations."
    )


    # -----------------------------------------------------------------
    # Model spatial tide height for every Landsat acquisition
    # -----------------------------------------------------------------

    tides = pixel_tides(
        data=ds,
        model="EOT20",
        directory=tide_model_dir,
        resample=True,
    )


    # -----------------------------------------------------------------
    # Align fixed long-term cutoff grid with this year's tide grid
    # -----------------------------------------------------------------

    cutoff_min = tide_cutoff_min.interp_like(
        tides.isel(time=0),
        method="linear",
    )

    cutoff_max = tide_cutoff_max.interp_like(
        tides.isel(time=0),
        method="linear",
    )


    # -----------------------------------------------------------------
    # Apply fixed long-term tidal acceptance window
    # -----------------------------------------------------------------

    accepted = (
        (tides >= cutoff_min)
        & (tides <= cutoff_max)
    )


    # Fraction of all pixel-observations that pass the filter
    accepted_fraction = float(
        accepted.mean().values
    )


    # Number of accepted observations at every pixel
    accepted_count_map = accepted.sum(
        dim="time"
    )


    # Typical number of accepted observations contributing to a pixel
    accepted_count = float(
        accepted_count_map.mean().values
    )


    print(
        f"Accepted tide pixels: "
        f"{accepted_fraction:.1%}"
    )

    print(
        f"Mean accepted observations per pixel: "
        f"{accepted_count:.1f}"
    )


    # -----------------------------------------------------------------
    # Mask tide values exactly as the annual composite would
    # -----------------------------------------------------------------

    accepted_tides = tides.where(
        accepted
    )


    # -----------------------------------------------------------------
    # Median contributing tide at each pixel for this year
    # -----------------------------------------------------------------

    annual_tide = accepted_tides.median(
        dim="time",
        skipna=True,
    )


    # -----------------------------------------------------------------
    # Summarise annual tide map to one value for QA
    #
    # Median = typical residual tide bias across Montrose
    # p10/p90 = spatial variation across Montrose
    # -----------------------------------------------------------------

    values = annual_tide.values.ravel()

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
            np.percentile(values, 10)
        )

        p90_tide = float(
            np.percentile(values, 90)
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


    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    results.append(
        {
            "year": year,
            "n_scenes": n_scenes,
            "accepted_count": accepted_count,
            "median_tide_m": median_tide,
            "p10_tide_m": p10_tide,
            "p90_tide_m": p90_tide,
            "accepted_fraction": accepted_fraction,
        }
    )


# ---------------------------------------------------------------------
# Convert results to DataFrame
# ---------------------------------------------------------------------

df = pd.DataFrame(results)

print()
print("Annual tide QA:")
print(df.to_string(index=False))


# ---------------------------------------------------------------------
# Save QA table
# ---------------------------------------------------------------------

csv_file = (
    "montrose_annual_tide_QA_1984_2025.csv"
)

df.to_csv(
    csv_file,
    index=False,
)

print(
    f"\nSaved {csv_file}"
)


# ---------------------------------------------------------------------
# Plot annual median contributing tide height
# ---------------------------------------------------------------------

valid = df.dropna(
    subset=[
        "median_tide_m",
        "accepted_count",
    ]
)


fig, ax = plt.subplots(
    figsize=(13, 6)
)


# ---------------------------------------------------------------------
# Line connecting annual tide values
# ---------------------------------------------------------------------

ax.plot(
    valid["year"],
    valid["median_tide_m"],
    linewidth=1.5,
    label="Median contributing tide",
)


# ---------------------------------------------------------------------
# Circle size proportional to number of accepted observations
#
# matplotlib's `s` parameter controls marker AREA.
# Multiplying accepted_count by a scale therefore makes circle area
# proportional to observation count.
# ---------------------------------------------------------------------

size_scale = 12

marker_sizes = (
    valid["accepted_count"]
    * size_scale
)

ax.scatter(
    valid["year"],
    valid["median_tide_m"],
    s=marker_sizes,
    zorder=3,
    label="Accepted Landsat observations",
)


# ---------------------------------------------------------------------
# Spatial 10–90% range
# ---------------------------------------------------------------------

ax.fill_between(
    valid["year"],
    valid["p10_tide_m"],
    valid["p90_tide_m"],
    alpha=0.2,
    label="Spatial 10–90% range",
)


# ---------------------------------------------------------------------
# Mean sea level reference
# ---------------------------------------------------------------------

ax.axhline(
    0,
    linestyle="--",
    linewidth=1,
    label="Mean sea level",
)


# ---------------------------------------------------------------------
# Main legend
# ---------------------------------------------------------------------

main_legend = ax.legend(
    loc="upper right",
)

ax.add_artist(
    main_legend
)


# ---------------------------------------------------------------------
# Circle-size legend
# ---------------------------------------------------------------------

count_examples = [
    int(valid["accepted_count"].min()),
    int(valid["accepted_count"].median()),
    int(valid["accepted_count"].max()),
]

# Remove duplicates
count_examples = sorted(
    set(count_examples)
)

size_handles = []

for count in count_examples:

    handle = ax.scatter(
        [],
        [],
        s=count * size_scale,
        label=f"{count} observations",
    )

    size_handles.append(
        handle
    )


ax.legend(
    handles=size_handles,
    title=(
        "Mean accepted\n"
        "observations per pixel"
    ),
    loc="lower right",
)


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

ax.set_xlabel(
    "Year"
)

ax.set_ylabel(
    "Median contributing tide height "
    "relative to mean sea level (m)"
)

ax.set_title(
    "Residual tidal sampling bias in annual Landsat composites\n"
    "Montrose Bay — EOT20, fixed 1984–2025 tidal window"
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


# ---------------------------------------------------------------------
# Save figure
# ---------------------------------------------------------------------

output_file = (
    "montrose_median_contributing_tide_1984_2025.png"
)

plt.savefig(
    output_file,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print(
    f"Saved {output_file}"
)

print("\nDone.")