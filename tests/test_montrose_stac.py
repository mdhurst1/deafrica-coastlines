from coastlines.stac import (
    load_water_index_stac,
    load_dem_stac,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystac_client


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

bbox = [-2.52, 56.70, -2.42, 56.76]

start_date = "1984-01-01"
end_date = "2025-12-31"

platforms = [
    "landsat-5",
    "landsat-7",
    "landsat-8",
    "landsat-9",
]

# ---------------------------------------------------------------------
# Load Copernicus DEM for terrain-shadow masking
# ---------------------------------------------------------------------

print("Loading Copernicus DEM...")

dem = load_dem_stac(
    bbox=bbox,
    output_crs="EPSG:32630",
    resolution=30,
)

print("\nDEM:")
print(dem)

# ---------------------------------------------------------------------
# Load full Landsat archive lazily
# ---------------------------------------------------------------------

print("Loading Montrose Landsat catalogue...")

ds = load_water_index_stac(
    bbox=bbox,
    datetime=f"{start_date}/{end_date}",
    platforms=platforms,
)

print("\nDataset:")
print(ds)


# ---------------------------------------------------------------------
# Count observations per year
# ---------------------------------------------------------------------

years = pd.DatetimeIndex(ds.time.values).year

unique_years, counts = np.unique(
    years,
    return_counts=True,
)

print("\nObservations per year:")

for year, count in zip(unique_years, counts):
    print(f"{year}: {count}")

print(f"\nTotal observations: {len(ds.time)}")


# ---------------------------------------------------------------------
# Plot observations per year
# ---------------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    unique_years,
    counts,
)

plt.xlabel("Year")
plt.ylabel("Number of Landsat observations")

plt.title(
    "Montrose Landsat observations per year "
    "(1984–2025)"
)

plt.xticks(
    unique_years[::2],
    rotation=45,
)

plt.tight_layout()

plt.savefig(
    "montrose_observations_per_year.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("\nSaved montrose_observations_per_year.png")


# ---------------------------------------------------------------------
# Find a low-cloud example scene using STAC metadata
#
# This avoids downloading every image in the full 1984–2025 archive
# just to determine which observation is clearest.
# ---------------------------------------------------------------------

print("\nSearching for lowest-cloud Landsat scene...")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime=f"{start_date}/{end_date}",
    query={
        "platform": {
            "in": platforms
        },
        "landsat:collection_category": {
            "eq": "T1"
        },
    },
)

items = search.item_collection()

if len(items) == 0:
    raise RuntimeError(
        "No Landsat scenes found for the Montrose test area."
    )


def cloud_cover(item):
    """Return scene cloud cover, treating missing values as 100%."""
    value = item.properties.get("eo:cloud_cover")

    if value is None:
        return 100.0

    return float(value)


# Exclude Landsat 7 SLC-off scenes from the example image
# (SLC failure occurred in 2003)
example_items = [
    item
    for item in items
    if not (
        item.properties.get("platform") == "landsat-7"
        and item.datetime.date() >= pd.Timestamp("2003-06-01").date()
    )
]

best_item = min(
    example_items,
    key=cloud_cover,
)

best_date = best_item.datetime.date()
best_platform = best_item.properties["platform"]
best_cloud = cloud_cover(best_item)

print("\nSelected example scene:")
print(f"  ID: {best_item.id}")
print(f"  Date: {best_date}")
print(f"  Platform: {best_platform}")
print(f"  Scene cloud cover: {best_cloud:.1f}%")


# ---------------------------------------------------------------------
# Load only the selected day
#
# This creates fresh Planetary Computer URLs immediately before
# downloading the example image.
# ---------------------------------------------------------------------

print("\nLoading example image WITHOUT terrain-shadow masking...")

example_ds_unmasked = load_water_index_stac(
    bbox=bbox,
    datetime=f"{best_date}/{best_date}",
    platforms=[best_platform],
    cloud_cover=100,
)


print("\nLoading example image WITH terrain-shadow masking...")

example_ds = load_water_index_stac(
    bbox=bbox,
    datetime=f"{best_date}/{best_date}",
    platforms=[best_platform],
    cloud_cover=100,
    dem=dem,
    mask_terrain_shadow=True,
    terrain_shadow_threshold=0.5,
    terrain_shadow_radius=1,
)

print("\nShadow-masked dataset:")
print(example_ds)

# ---------------------------------------------------------------------
# Terrain-shadow diagnostics
# ---------------------------------------------------------------------

print("\nSolar geometry:")

print(
    "  Sun elevation:",
    example_ds.sun_elevation.values,
)

print(
    "  Sun azimuth:",
    example_ds.sun_azimuth.values,
)


shadow = (
    example_ds
    .terrain_shadow
    .isel(time=0)
    .compute()
)

shadow_pixels = int(
    shadow.sum().values
)

total_pixels = shadow.size

shadow_percent = (
    100.0 * shadow_pixels / total_pixels
)

print("\nTerrain-shadow mask:")
print(f"  Shadow pixels: {shadow_pixels}")
print(f"  Total pixels: {total_pixels}")
print(f"  Shadowed area: {shadow_percent:.2f}%")

# ---------------------------------------------------------------------
# Compute and plot MNDWI for the example scene
# ---------------------------------------------------------------------

print("\nComputing example MNDWI...")

example_unmasked = (
    example_ds_unmasked
    .mndwi
    .isel(time=0)
    .compute()
)

example_masked = (
    example_ds
    .mndwi
    .isel(time=0)
    .compute()
)

shadow = (
    example_ds
    .terrain_shadow
    .isel(time=0)
    .compute()
)

valid_before = np.isfinite(
    example_unmasked.values
)

valid_after = np.isfinite(
    example_masked.values
)

newly_masked = (
    valid_before
    & ~valid_after
)

print(
    "Valid MNDWI pixels removed by terrain shadow:",
    int(newly_masked.sum()),
)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 7),
)

example_unmasked.plot(
    ax=axes[0],
    vmin=-1,
    vmax=1,
    cmap="RdBu",
)

axes[0].set_title(
    "MNDWI - no terrain-shadow mask"
)

shadow.plot(
    ax=axes[1],
    cmap="gray",
    add_colorbar=False,
)

axes[1].set_title(
    "Terrain-shadow mask"
)

example_masked.plot(
    ax=axes[2],
    vmin=-1,
    vmax=1,
    cmap="RdBu",
)

axes[2].set_title(
    "MNDWI - terrain-shadow masked"
)

fig.suptitle(
    f"Montrose terrain-shadow test\n"
    f"{best_date} - {best_platform}"
)

plt.tight_layout()

plt.savefig(
    "montrose_terrain_shadow_test.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved montrose_mndwi_example.png")

print("\nDone.")