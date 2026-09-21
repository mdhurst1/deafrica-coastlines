from coastlines.stac import load_water_index_stac

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

print("\nLoading example image...")

example_ds = load_water_index_stac(
    bbox=bbox,
    datetime=f"{best_date}/{best_date}",
    platforms=[best_platform],
    cloud_cover=100,
)

print(example_ds)


# ---------------------------------------------------------------------
# Compute and plot MNDWI for the example scene
# ---------------------------------------------------------------------

print("\nComputing example MNDWI...")

example = (
    example_ds
    .mndwi
    .isel(time=0)
    .compute()
)

plt.figure(figsize=(8, 8))

example.plot(
    vmin=-1,
    vmax=1,
    cmap="RdBu",
)

plt.title(
    f"Montrose MNDWI\n"
    f"{best_date} — {best_platform} "
    f"({best_cloud:.1f}% scene cloud)"
)

plt.tight_layout()

plt.savefig(
    "montrose_mndwi_example.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print("Saved montrose_mndwi_example.png")


# ---------------------------------------------------------------------
# Finished
# ---------------------------------------------------------------------

print("\nDone.")