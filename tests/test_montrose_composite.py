import matplotlib.pyplot as plt

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

year = 2020

tide_model_dir = "/home/mh322u/tide_models"

platforms = [
    "landsat-7",
    "landsat-8",
]


# ---------------------------------------------------------------------
# Load Landsat water-index stack
# ---------------------------------------------------------------------

print(f"Loading Montrose Landsat data for {year}...")

ds = load_water_index_stac(
    bbox=bbox,
    datetime=f"{year}-01-01/{year}-12-31",
    platforms=platforms,
)

print("\nInput dataset:")
print(ds)

print(
    f"\nInitial Landsat observations: "
    f"{ds.sizes['time']}"
)


# ---------------------------------------------------------------------
# Model tides
# ---------------------------------------------------------------------

print("\nModelling low-resolution tides...")

tides_lowres = pixel_tides(
    data=ds,
    model="EOT20",
    directory=tide_model_dir,
    resample=False,
)

print("\nModelling tides on Landsat grid...")

tides_highres = pixel_tides(
    data=ds,
    model="EOT20",
    directory=tide_model_dir,
    resample=True,
)

# Keep old DEA Coastlines variable name
ds["tide_m"] = tides_highres


# ---------------------------------------------------------------------
# Calculate tidal acceptance window
# ---------------------------------------------------------------------

print("\nCalculating tidal cutoffs...")

tide_cutoff_min, tide_cutoff_max = tide_cutoffs(
    ds=ds,
    tides_lowres=tides_lowres,
    tide_centre=0.0,
)

print(
    "Mean tidal window: "
    f"{float(tide_cutoff_min.mean().values):.2f} m "
    "to "
    f"{float(tide_cutoff_max.mean().values):.2f} m"
)


# ---------------------------------------------------------------------
# Apply tidal filter
# ---------------------------------------------------------------------

print("\nApplying tidal filtering...")

tidal_ds = load_tidal_subset(
    year_ds=ds,
    tide_cutoff_min=tide_cutoff_min,
    tide_cutoff_max=tide_cutoff_max,
)

print(
    f"Observations remaining after tidal filtering: "
    f"{tidal_ds.sizes['time']}"
)


# ---------------------------------------------------------------------
# Generate annual DEA Coastlines composite
# ---------------------------------------------------------------------

print("\nGenerating annual composite...")

composite = tidal_composite(
    year_ds=tidal_ds,
    label=year,
    label_dim="year",
    output_dir=".",
    export_geotiff=False,
)

print("\nAnnual composite:")
print(composite)

print("\nComposite variables:")

for variable in composite.data_vars:
    print(f"  {variable}")


# ---------------------------------------------------------------------
# Remove single year dimension for plotting
# ---------------------------------------------------------------------

annual = composite.sel(year=year)


# ---------------------------------------------------------------------
# Basic diagnostics
# ---------------------------------------------------------------------

count_min = int(annual["count"].min().values)
count_max = int(annual["count"].max().values)
count_mean = float(annual["count"].mean().values)

print("\nValid observation count:")
print(f"  Minimum: {count_min}")
print(f"  Maximum: {count_max}")
print(f"  Mean:    {count_mean:.1f}")

print("\nMNDWI:")
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


# ---------------------------------------------------------------------
# Plot annual MNDWI composite
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 8))

annual.mndwi.plot(
    ax=ax,
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)

ax.set_title(
    f"Montrose annual median MNDWI — {year}\n"
    "Tidally filtered Landsat composite"
)

plt.tight_layout()

plt.savefig(
    f"montrose_mndwi_composite_{year}.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"\nSaved montrose_mndwi_composite_{year}.png"
)


# ---------------------------------------------------------------------
# Four-panel QA figure
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(14, 12),
)

# Median MNDWI
annual.mndwi.plot(
    ax=axes[0, 0],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)

axes[0, 0].set_title(
    "Median MNDWI"
)


# Valid observation count
annual["count"].plot(
    ax=axes[0, 1],
    cmap="viridis",
)

axes[0, 1].set_title(
    "Valid observation count"
)


# MNDWI standard deviation
annual.stdev.plot(
    ax=axes[1, 0],
    cmap="magma",
)

axes[1, 0].set_title(
    "MNDWI standard deviation"
)


# Median tide height
annual.tide_m.plot(
    ax=axes[1, 1],
    cmap="RdBu",
)

axes[1, 1].set_title(
    "Median contributing tide height"
)


fig.suptitle(
    f"Montrose DEA Coastlines annual composite — {year}",
    fontsize=16,
)

plt.tight_layout()

plt.savefig(
    f"montrose_composite_QA_{year}.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved montrose_composite_QA_{year}.png"
)

print("\nMNDWI composite range:")
print(
    float(annual.mndwi.min().values),
    float(annual.mndwi.max().values),
)

print("\nMNDWI stdev range:")
print(
    float(annual.stdev.min().values),
    float(annual.stdev.max().values),
)

# Number of satellite acquisitions accepted by the tide filter
tide_bool = (
    (ds.tide_m >= tide_cutoff_min)
    & (ds.tide_m <= tide_cutoff_max)
)

accepted_tide_count = tide_bool.sum(dim="time")

fig, ax = plt.subplots(figsize=(8, 7))

accepted_tide_count.plot(
    ax=ax,
    cmap="viridis",
)

ax.set_title(
    "Number of Landsat acquisitions accepted by tidal filter\n"
    "Montrose Bay — 2020"
)

plt.tight_layout()

plt.savefig(
    "montrose_tide_acceptance_count_2020.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("\nAnnual composite complete.")