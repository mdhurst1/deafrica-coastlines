import matplotlib.pyplot as plt

from eo_tides.eo import pixel_tides

from coastlines.stac import load_water_index_stac
from coastlines.raster import tide_cutoffs, load_tidal_subset


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

bbox = [-2.52, 56.70, -2.42, 56.76]

tide_model_dir = "/home/mh322u/tide_models"

start_date = "2020-01-01"
end_date = "2020-12-31"

platforms = [
    "landsat-7",
    "landsat-8",
]


# ---------------------------------------------------------------------
# Load Montrose Landsat stack
# ---------------------------------------------------------------------

print("Loading Montrose Landsat observations...")

ds = load_water_index_stac(
    bbox=bbox,
    datetime=f"{start_date}/{end_date}",
    platforms=platforms,
)

print("\nSatellite dataset:")
print(ds)

print("\nSatellite dimensions:")
print(
    f"time={len(ds.time)}, "
    f"y={len(ds.y)}, "
    f"x={len(ds.x)}"
)

print("\nSpatial metadata:")
print(ds.odc.geobox)


# ---------------------------------------------------------------------
# Model LOW-resolution tides
# ---------------------------------------------------------------------

print("\nModelling LOW-resolution pixel tides...")

tides_lowres = pixel_tides(
    data=ds,
    model="EOT20",
    directory=tide_model_dir,
    resample=False,
)

print("\nLow-resolution tide dataset:")
print(tides_lowres)


# ---------------------------------------------------------------------
# Model tides resampled to Landsat grid
# ---------------------------------------------------------------------

print("\nModelling HIGH-resolution pixel tides...")

tides_highres = pixel_tides(
    data=ds,
    model="EOT20",
    directory=tide_model_dir,
    resample=True,
)

print("\nHigh-resolution tide dataset:")
print(tides_highres)


# ---------------------------------------------------------------------
# Check dimensions
# ---------------------------------------------------------------------

assert len(tides_highres.time) == len(ds.time)
assert len(tides_highres.y) == len(ds.y)
assert len(tides_highres.x) == len(ds.x)

print(
    "\nHigh-resolution tide dimensions match "
    "the Landsat dataset."
)


# ---------------------------------------------------------------------
# Attach tide heights to Landsat dataset
#
# Keep the name tide_m because that is what the existing
# DEA Coastlines functions expect.
# ---------------------------------------------------------------------

ds["tide_m"] = tides_highres

print("\nDataset after adding tide_m:")
print(ds)


# ---------------------------------------------------------------------
# Basic tide diagnostics
# ---------------------------------------------------------------------

tide_min = float(tides_highres.min().values)
tide_max = float(tides_highres.max().values)
tide_mean = float(tides_highres.mean().values)

print("\nModelled tide statistics:")
print(f"  Minimum: {tide_min:.2f} m")
print(f"  Maximum: {tide_max:.2f} m")
print(f"  Mean:    {tide_mean:.2f} m")


# ---------------------------------------------------------------------
# Calculate DEA Coastlines tidal cutoffs
#
# This uses the ORIGINAL tide_cutoffs() function from raster.py.
# ---------------------------------------------------------------------

print("\nCalculating DEA Coastlines tidal cutoffs...")

tide_cutoff_min, tide_cutoff_max = tide_cutoffs(
    ds=ds,
    tides_lowres=tides_lowres,
    tide_centre=0.0,
)

print("\nTidal cutoff arrays:")
print("Minimum cutoff:")
print(tide_cutoff_min)

print("\nMaximum cutoff:")
print(tide_cutoff_max)


# ---------------------------------------------------------------------
# Inspect which Landsat observations pass the tidal filter
# ---------------------------------------------------------------------

print("\nChecking tidal acceptance for each Landsat observation...")

tide_bool = (
    (ds.tide_m >= tide_cutoff_min)
    & (ds.tide_m <= tide_cutoff_max)
)

# Percentage of pixels in each observation that fall inside
# the permitted tidal window
retained_fraction = (
    tide_bool
    .mean(dim=("x", "y"))
    .compute()
)

print("\nTidal acceptance by observation:")

for time, fraction in zip(
    ds.time.values,
    retained_fraction.values,
):
    print(
        f"  {str(time)[:19]}: "
        f"{fraction * 100:.1f}% of pixels accepted"
    )


# ---------------------------------------------------------------------
# Apply the ORIGINAL DEA Coastlines tidal filtering function
# ---------------------------------------------------------------------

print("\nApplying DEA Coastlines tidal filter...")

tidal_ds = load_tidal_subset(
    year_ds=ds,
    tide_cutoff_min=tide_cutoff_min,
    tide_cutoff_max=tide_cutoff_max,
)

print("\nTidally filtered dataset:")
print(tidal_ds)

print(
    f"\nObservations before filtering: {ds.sizes['time']}"
)

print(
    f"Observations remaining after filtering: "
    f"{tidal_ds.sizes['time']}"
)

# ---------------------------------------------------------------------
# Summarise cutoff values
# ---------------------------------------------------------------------

cutoff_min_mean = float(
    tide_cutoff_min.mean().values
)

cutoff_max_mean = float(
    tide_cutoff_max.mean().values
)

cutoff_min_global = float(
    tide_cutoff_min.min().values
)

cutoff_max_global = float(
    tide_cutoff_max.max().values
)

print("\nTidal cutoff summary:")

print(
    f"  Mean lower cutoff: "
    f"{cutoff_min_mean:.2f} m"
)

print(
    f"  Mean upper cutoff: "
    f"{cutoff_max_mean:.2f} m"
)

print(
    f"  Lowest lower cutoff: "
    f"{cutoff_min_global:.2f} m"
)

print(
    f"  Highest upper cutoff: "
    f"{cutoff_max_global:.2f} m"
)


# ---------------------------------------------------------------------
# Choose an example Landsat observation
# ---------------------------------------------------------------------

example_idx = 0

example_time = ds.time.isel(
    time=example_idx
).values

print(
    f"\nPlotting example timestep: "
    f"{example_time}"
)


# ---------------------------------------------------------------------
# Plot low-resolution tide surface
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 7)
)

tides_lowres.isel(
    time=example_idx
).plot(
    ax=ax,
    cmap="RdBu",
)

ax.set_title(
    "EOT20 low-resolution tide surface\n"
    f"Montrose Bay — {example_time}"
)

plt.tight_layout()

plt.savefig(
    "montrose_tides_lowres.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved montrose_tides_lowres.png"
)


# ---------------------------------------------------------------------
# Plot high-resolution tide surface
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 7)
)

tides_highres.isel(
    time=example_idx
).plot(
    ax=ax,
    cmap="RdBu",
)

ax.set_title(
    "EOT20 tide surface resampled to Landsat grid\n"
    f"Montrose Bay — {example_time}"
)

plt.tight_layout()

plt.savefig(
    "montrose_tides_highres.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved montrose_tides_highres.png"
)


# ---------------------------------------------------------------------
# Plot tidal cutoff surfaces
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
    "DEA Coastlines tidal acceptance window — Montrose",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    "montrose_tide_cutoffs.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved montrose_tide_cutoffs.png"
)


# ---------------------------------------------------------------------
# Plot MNDWI alongside tide surface
# ---------------------------------------------------------------------

mndwi_example = (
    ds.mndwi
    .isel(time=example_idx)
    .compute()
)

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(14, 6),
)

mndwi_example.plot(
    ax=axes[0],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)

axes[0].set_title(
    "Landsat MNDWI"
)

tides_highres.isel(
    time=example_idx
).plot(
    ax=axes[1],
    cmap="RdBu",
)

axes[1].set_title(
    "EOT20 tide height"
)

fig.suptitle(
    f"Montrose Bay — {example_time}",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    "montrose_mndwi_and_tides.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved montrose_mndwi_and_tides.png"
)

# ---------------------------------------------------------------------
# Plot number of tidally accepted observations
# ---------------------------------------------------------------------

valid_count = tidal_ds.mndwi.count(dim="time")

fig, ax = plt.subplots(figsize=(8, 7))

valid_count.plot(
    ax=ax,
    cmap="viridis",
)

ax.set_title(
    "Tidally accepted Landsat observations\n"
    "Montrose Bay — 2020"
)

plt.tight_layout()

plt.savefig(
    "montrose_tidal_observation_count.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved montrose_tidal_observation_count.png"
)

# ---------------------------------------------------------------------
# Compare MNDWI before and after tidal filtering
# ---------------------------------------------------------------------

if tidal_ds.sizes["time"] > 0:

    example_time = tidal_ds.time.values[0]

    print(
        f"\nComparing before/after filtering for "
        f"{example_time}"
    )

    before = (
        ds.mndwi
        .sel(time=example_time)
        .compute()
    )

    after = tidal_ds.mndwi.sel(
        time=example_time
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 6),
    )

    before.plot(
        ax=axes[0],
        cmap="RdBu",
        vmin=-1,
        vmax=1,
    )

    axes[0].set_title(
        "Before tidal filtering"
    )

    after.plot(
        ax=axes[1],
        cmap="RdBu",
        vmin=-1,
        vmax=1,
    )

    axes[1].set_title(
        "After tidal filtering"
    )

    fig.suptitle(
        f"Montrose MNDWI — {str(example_time)[:10]}"
    )

    plt.tight_layout()

    plt.savefig(
        "montrose_mndwi_tidal_filter.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved montrose_mndwi_tidal_filter.png"
    )

print("\nPixel tide + cutoff test complete.")