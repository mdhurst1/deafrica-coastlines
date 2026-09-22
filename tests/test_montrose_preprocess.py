from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

from coastlines.vector import (
    load_rasters,
    contours_preprocess,
)


# =====================================================================
# SETTINGS
# =====================================================================

raster_path = "outputs/dea_rasters"

raster_version = "scotland_v0.1"

study_area = "montrose"

start_year = 1988
end_year = 2025

water_index = "mndwi"

index_threshold = 0.0


output_dir = Path(
    "outputs/montrose_vector_QA"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# LOAD RASTER STACK
# =====================================================================

print()
print("=" * 70)
print("LOADING DEA RASTERS")
print("=" * 70)


yearly_ds, gapfill_ds = load_rasters(
    path=raster_path,
    raster_version=raster_version,
    study_area=study_area,
    water_index=water_index,
    start_year=start_year,
    end_year=end_year,
)


print(
    f"Years: {yearly_ds.year.values}"
)

print(
    f"Variables: {list(yearly_ds.data_vars)}"
)


# =====================================================================
# CREATE NORTH SEA OCEAN SEED POINTS
#
# DEA Africa uses tidal-model points known to lie in marine water.
#
# For this Montrose test we create equivalent seed points along the
# eastern edge of the Landsat AOI, which lies offshore in the North Sea.
# =====================================================================

x_values = yearly_ds.x.values
y_values = yearly_ds.y.values


# Put points a few Landsat pixels inside the eastern edge.
#
# Using x[-3] rather than the exact edge avoids any edge effects.
ocean_x = float(
    x_values[-3]
)


# Generate approximately 12 points along the N-S extent.
n_points = 12


indices = [
    int(i)
    for i in
    __import__("numpy").linspace(
        0,
        len(y_values) - 1,
        n_points,
    )
]


ocean_points = [
    Point(
        ocean_x,
        float(y_values[i]),
    )
    for i in indices
]


# Obtain CRS from rioxarray
crs = yearly_ds.rio.crs


tide_points_gdf = gpd.GeoDataFrame(
    geometry=ocean_points,
    crs=crs,
)


print()
print(
    f"Created {len(tide_points_gdf)} "
    f"North Sea ocean seed points."
)

print(
    f"Ocean seed x coordinate: "
    f"{ocean_x:.1f}"
)

print(
    f"CRS: {crs}"
)


# =====================================================================
# SAVE SEED POINTS FOR QA
# =====================================================================

seed_file = (
    output_dir
    / "montrose_ocean_seed_points.geojson"
)


tide_points_gdf.to_file(
    seed_file,
    driver="GeoJSON",
)


print(
    f"Saved ocean seed points: "
    f"{seed_file}"
)


# =====================================================================
# RUN DEA AFRICA PREPROCESSING
# =====================================================================

print()
print("=" * 70)
print("RUNNING DEA AFRICA CONTOUR PREPROCESSING")
print("=" * 70)


masked_ds, certainty_masks = contours_preprocess(
    yearly_ds=yearly_ds,
    gapfill_ds=gapfill_ds,
    water_index=water_index,
    index_threshold=index_threshold,
    tide_points_gdf=tide_points_gdf,

    # DEA Africa default = 33 pixels ≈ 1 km
    buffer_pixels=33,

    # We do not currently have ESA WorldCover
    # indexed in our local ODC Datacube.
    mask_landcover=False,

    # Keep the Africa NDWI cleaning step.
    mask_ndwi=True,

    # Keep temporal-contiguity cleaning.
    mask_temporal=True,

    # No manual edits for Montrose yet.
    mask_modifications=None,
)


# =====================================================================
# REPORT
# =====================================================================

print()
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)


print(masked_ds)


print()
print(
    f"Number of annual certainty masks: "
    f"{len(certainty_masks)}"
)


# =====================================================================
# QA FIGURE
# =====================================================================

qa_year = 2020


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 7),
)


# ---------------------------------------------------------------------
# Original annual MNDWI
# ---------------------------------------------------------------------

yearly_ds[
    water_index
].sel(
    year=qa_year
).plot(
    ax=axes[0],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)


axes[0].set_title(
    f"{qa_year} annual MNDWI"
)


# Add ocean seed points
axes[0].scatter(
    tide_points_gdf.geometry.x,
    tide_points_gdf.geometry.y,
    marker="x",
    s=40,
)


# ---------------------------------------------------------------------
# Final DEA-preprocessed MNDWI
# ---------------------------------------------------------------------

masked_ds.sel(
    year=qa_year
).plot(
    ax=axes[1],
    cmap="RdBu",
    vmin=-1,
    vmax=1,
)


axes[1].set_title(
    f"{qa_year} DEA Africa masked MNDWI"
)


for ax in axes:

    ax.set_aspect(
        "equal"
    )


plt.tight_layout()


qa_file = (
    output_dir
    / "montrose_preprocess_2020.png"
)


plt.savefig(
    qa_file,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print()
print(
    f"Saved QA plot: "
    f"{qa_file}"
)

print()
print("Done.")