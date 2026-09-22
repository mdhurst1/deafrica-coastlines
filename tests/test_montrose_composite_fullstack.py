import matplotlib.pyplot as plt
import xarray as xr

from eo_tides.eo import pixel_tides

from coastlines.stac import load_water_index_stac
from coastlines.raster import (
    tide_cutoffs,
    load_tidal_subset,
    tidal_composite,
)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

bbox = [-2.52, 56.70, -2.42, 56.76]

cutoff_start_year = 1984
cutoff_end_year = 2025

composite_year = 2020

tide_model_dir = "/home/mh322u/tide_models"

platforms = [
    "landsat-5",
    "landsat-7",
    "landsat-8",
    "landsat-9",
]


# =====================================================================
# PART 1
# Calculate long-term tidal cutoffs from the entire Landsat record
# =====================================================================

print(
    f"Loading Landsat catalogue for "
    f"{cutoff_start_year}–{cutoff_end_year}..."
)

# This remains lazy: we are primarily using the acquisition times
# and geospatial metadata, not downloading all the imagery.
ds_full = load_water_index_stac(
    bbox=bbox,
    datetime=(
        f"{cutoff_start_year}-01-01/"
        f"{cutoff_end_year}-12-31"
    ),
    platforms=platforms,
)

print(
    f"\nFull Landsat record contains "
    f"{ds_full.sizes['time']} observations."
)


# ---------------------------------------------------------------------
# Model LOW-RESOLUTION tides for every Landsat acquisition time
#
# We only need the low-resolution tide grid to calculate the
# long-term tidal acceptance window.
# ---------------------------------------------------------------------

print(
    "\nModelling low-resolution tides for "
    "the full Landsat record..."
)

tides_lowres_full = pixel_tides(
    data=ds_full,
    model="EOT20",
    directory=tide_model_dir,
    resample=False,
)

print("\nFull-record low-resolution tides:")
print(tides_lowres_full)


# =====================================================================
# PART 2
# Load the year we actually want to composite
# =====================================================================

print(
    f"\nLoading Landsat imagery for "
    f"{composite_year}..."
)

ds_year = load_water_index_stac(
    bbox=bbox,
    datetime=(
        f"{composite_year}-01-01/"
        f"{composite_year}-12-31"
    ),
    platforms=platforms,
)

print(
    f"\n{composite_year} contains "
    f"{ds_year.sizes['time']} Landsat observations."
)


# ---------------------------------------------------------------------
# Calculate long-term tidal cutoffs
#
# tides_lowres_full supplies the 1984–2025 tide range.
#
# ds_year is deliberately supplied as the target spatial grid so that
# the resulting cutoff rasters line up exactly with the 2020 imagery.
# ---------------------------------------------------------------------

print(
    "\nCalculating long-term tidal cutoffs "
    f"from {cutoff_start_year}–{cutoff_end_year}..."
)

tide_cutoff_min, tide_cutoff_max = tide_cutoffs(
    ds=ds_year,
    tides_lowres=tides_lowres_full,
    tide_centre=0.0,
)

print("\nLong-term tidal cutoff statistics:")

print(
    f"  Mean lower cutoff: "
    f"{float(tide_cutoff_min.mean().values):.2f} m"
)

print(
    f"  Mean upper cutoff: "
    f"{float(tide_cutoff_max.mean().values):.2f} m"
)

print(
    f"  Minimum lower cutoff: "
    f"{float(tide_cutoff_min.min().values):.2f} m"
)

print(
    f"  Maximum upper cutoff: "
    f"{float(tide_cutoff_max.max().values):.2f} m"
)


# ---------------------------------------------------------------------
# Save the long-term cutoff surfaces
#
# These can be reused for every annual Montrose composite rather than
# recalculating the full 1984–2025 tide record each time.
# ---------------------------------------------------------------------

cutoff_ds = xr.Dataset(
    {
        "tide_cutoff_min": tide_cutoff_min,
        "tide_cutoff_max": tide_cutoff_max,
    }
)

cutoff_file = (
    f"montrose_tidal_cutoffs_"
    f"{cutoff_start_year}_{cutoff_end_year}.nc"
)

cutoff_ds.to_netcdf(cutoff_file)

print(
    f"\nSaved long-term tidal cutoffs to "
    f"{cutoff_file}"
)


# ---------------------------------------------------------------------
# Plot long-term tidal cutoff surfaces
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 6),
)

tide_cutoff_min.plot(
    ax=axes[0],
    cmap="RdBu",
)

axes[0].set_title(
    "Lower tidal cutoff"
)

tide_cutoff_max.plot(
    ax=axes[1],
    cmap="RdBu",
)

axes[1].set_title(
    "Upper tidal cutoff"
)

fig.suptitle(
    "Montrose long-term Landsat tidal acceptance window\n"
    f"{cutoff_start_year}–{cutoff_end_year}",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    "montrose_longterm_tidal_cutoffs.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved montrose_longterm_tidal_cutoffs.png"
)


# =====================================================================
# PART 3
# Model tides for the 2020 imagery
# =====================================================================

print(
    f"\nModelling pixel tides for "
    f"{composite_year} imagery..."
)

tides_year = pixel_tides(
    data=ds_year,
    model="EOT20",
    directory=tide_model_dir,
    resample=True,
)

# Preserve original DEA Coastlines variable name
ds_year["tide_m"] = tides_year


# =====================================================================
# PART 4
# Apply the FIXED 1984–2025 tidal window to 2020
# =====================================================================

print(
    "\nApplying long-term tidal acceptance "
    f"window to {composite_year}..."
)

# Useful QA mask before loading the imagery
tide_bool = (
    (ds_year.tide_m >= tide_cutoff_min)
    & (ds_year.tide_m <= tide_cutoff_max)
)

accepted_count = tide_bool.sum(
    dim="time"
)

print("\nAccepted acquisitions per pixel:")

print(
    f"  Minimum: "
    f"{int(accepted_count.min().values)}"
)

print(
    f"  Maximum: "
    f"{int(accepted_count.max().values)}"
)

print(
    f"  Mean: "
    f"{float(accepted_count.mean().values):.1f}"
)


# ---------------------------------------------------------------------
# Apply original DEA Coastlines tidal filtering
# ---------------------------------------------------------------------

tidal_ds = load_tidal_subset(
    year_ds=ds_year,
    tide_cutoff_min=tide_cutoff_min,
    tide_cutoff_max=tide_cutoff_max,
)

print(
    f"\nObservations before tidal filtering: "
    f"{ds_year.sizes['time']}"
)

print(
    f"Observations remaining after filtering: "
    f"{tidal_ds.sizes['time']}"
)


# =====================================================================
# PART 5
# Generate annual composite
# =====================================================================

print(
    f"\nGenerating {composite_year} "
    "annual composite..."
)

composite = tidal_composite(
    year_ds=tidal_ds,
    label=composite_year,
    label_dim="year",
    output_dir=".",
    export_geotiff=False,
)

annual = composite.sel(
    year=composite_year
)

print("\nAnnual composite:")
print(annual)


# ---------------------------------------------------------------------
# Composite diagnostics
# ---------------------------------------------------------------------

print("\nValid observation count:")

print(
    f"  Minimum: "
    f"{int(annual['count'].min().values)}"
)

print(
    f"  Maximum: "
    f"{int(annual['count'].max().values)}"
)

print(
    f"  Mean: "
    f"{float(annual['count'].mean().values):.1f}"
)


print("\nMNDWI range:")

print(
    f"  Minimum: "
    f"{float(annual.mndwi.min().values):.3f}"
)

print(
    f"  Maximum: "
    f"{float(annual.mndwi.max().values):.3f}"
)


print("\nMNDWI standard deviation:")

print(
    f"  Mean: "
    f"{float(annual.stdev.mean().values):.3f}"
)

print(
    f"  Maximum: "
    f"{float(annual.stdev.max().values):.3f}"
)


# =====================================================================
# PART 6
# QA plots
# =====================================================================

# ---------------------------------------------------------------------
# Tidal acceptance count
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 7)
)

accepted_count.plot(
    ax=ax,
    cmap="viridis",
)

ax.set_title(
    f"Landsat acquisitions accepted using "
    f"{cutoff_start_year}–{cutoff_end_year} tidal window\n"
    f"Montrose Bay — {composite_year}"
)

plt.tight_layout()

plt.savefig(
    f"montrose_tide_acceptance_count_{composite_year}.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ---------------------------------------------------------------------
# Four-panel annual composite QA
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(14, 12),
)


annual.mndwi.plot(
    ax=axes[0, 0],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)

axes[0, 0].set_title(
    "Median MNDWI"
)


annual["count"].plot(
    ax=axes[0, 1],
    cmap="viridis",
)

axes[0, 1].set_title(
    "Valid observation count"
)


annual.stdev.plot(
    ax=axes[1, 0],
    cmap="magma",
)

axes[1, 0].set_title(
    "MNDWI standard deviation"
)


annual.tide_m.plot(
    ax=axes[1, 1],
    cmap="RdBu",
)

axes[1, 1].set_title(
    "Median contributing tide height"
)


fig.suptitle(
    f"Montrose DEA Coastlines annual composite — "
    f"{composite_year}\n"
    f"Tidal cutoffs derived from "
    f"{cutoff_start_year}–{cutoff_end_year}",
    fontsize=15,
)

plt.tight_layout()

plt.savefig(
    f"montrose_composite_longterm_tides_"
    f"{composite_year}.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    f"\nSaved "
    f"montrose_composite_longterm_tides_"
    f"{composite_year}.png"
)

print("\nDone.")