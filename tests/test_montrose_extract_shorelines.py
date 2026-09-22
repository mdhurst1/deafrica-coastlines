from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from shapely.geometry import Point

from dea_tools.spatial import subpixel_contours

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
    "outputs/montrose_vectors"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


qa_dir = Path(
    "outputs/montrose_vector_QA"
)

qa_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# LOAD RASTERS
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
# =====================================================================

x_values = yearly_ds.x.values
y_values = yearly_ds.y.values


ocean_x = float(
    x_values[-3]
)


n_points = 12


indices = np.linspace(
    0,
    len(y_values) - 1,
    n_points,
).astype(int)


ocean_points = [
    Point(
        ocean_x,
        float(y_values[i]),
    )
    for i in indices
]


crs = yearly_ds.rio.crs


if crs is None:

    raise ValueError(
        "Raster stack has no CRS."
    )


tide_points_gdf = gpd.GeoDataFrame(
    geometry=ocean_points,
    crs=crs,
)


print()
print(
    f"Created {len(tide_points_gdf)} "
    f"North Sea seed points."
)


# =====================================================================
# PREPROCESS
# =====================================================================

print()
print("=" * 70)
print("RUNNING DEA AFRICA PREPROCESSING")
print("=" * 70)


masked_ds, certainty_masks = contours_preprocess(
    yearly_ds=yearly_ds,
    gapfill_ds=gapfill_ds,
    water_index=water_index,
    index_threshold=index_threshold,
    tide_points_gdf=tide_points_gdf,
    buffer_pixels=33,

    # No local ODC WorldCover product yet
    mask_landcover=False,

    mask_ndwi=True,
    mask_temporal=True,
    mask_modifications=None,
)


print()
print(
    "Preprocessing complete."
)


# =====================================================================
# EXTRACT SUBPIXEL SHORELINES
# =====================================================================

print()
print("=" * 70)
print("EXTRACTING SUBPIXEL SHORELINES")
print("=" * 70)


contours_gdf = subpixel_contours(
    da=masked_ds,
    z_values=index_threshold,
    min_vertices=10,
    dim="year",
)


print()
print(
    f"Extracted {len(contours_gdf)} "
    f"shoreline features."
)


# =====================================================================
# NORMALISE YEAR FIELD
# =====================================================================

# DEA's production code sets year as the index.
if "year" in contours_gdf.columns:

    contours_gdf["year"] = (
        contours_gdf["year"]
        .astype(int)
    )

    contours_gdf = contours_gdf.set_index(
        "year"
    )


else:

    # Defensive case if year is already index-like
    contours_gdf.index = (
        contours_gdf.index
        .astype(int)
    )

    contours_gdf.index.name = "year"


print()
print(
    "Years extracted:"
)

print(
    sorted(
        contours_gdf.index.unique()
    )
)


# =====================================================================
# REMOVE EMPTY GEOMETRIES
# =====================================================================

contours_gdf = contours_gdf[
    ~contours_gdf.geometry.is_empty
    & contours_gdf.geometry.notnull()
]


# =====================================================================
# ADD BASIC ATTRIBUTES
# =====================================================================

contours_gdf["tide_datum"] = (
    "0 m model MSL"
)


# =====================================================================
# SAVE COMPLETE SHORELINE DATASET
# =====================================================================

gpkg_file = (
    output_dir
    / "montrose_annual_shorelines_1988_2025.gpkg"
)


geojson_file = (
    output_dir
    / "montrose_annual_shorelines_1988_2025.geojson"
)


# GeoPackage in native projected CRS
contours_gdf.reset_index().to_file(
    gpkg_file,
    layer="annual_shorelines",
    driver="GPKG",
)


# GeoJSON in WGS84
contours_gdf.reset_index().to_crs(
    "EPSG:4326"
).to_file(
    geojson_file,
    driver="GeoJSON",
)


print()
print(
    f"Saved GeoPackage: {gpkg_file}"
)

print(
    f"Saved GeoJSON: {geojson_file}"
)


# =====================================================================
# SUMMARY BY YEAR
# =====================================================================

summary = (
    contours_gdf
    .groupby(level="year")
    .agg(
        n_features=("geometry", "count")
    )
)


summary["total_length_m"] = (
    contours_gdf
    .groupby(level="year")
    .geometry
    .apply(
        lambda g: g.length.sum()
    )
)


summary_file = (
    output_dir
    / "montrose_annual_shoreline_summary.csv"
)


summary.to_csv(
    summary_file
)


print()
print(
    summary
)


print()
print(
    f"Saved summary: {summary_file}"
)


# =====================================================================
# IMPROVED QA FIGURE
#
# Two panels:
#   1) Full context
#   2) Zoomed section of beach
#
# All shorelines are coloured by year using the magma colour ramp.
# =====================================================================

print()
print("=" * 70)
print("CREATING IMPROVED QA FIGURE")
print("=" * 70)

# Work in the native projected CRS for plotting / zooming
shorelines_plot = contours_gdf.reset_index().copy()

# ---------------------------------------------------------------------
# Choose a zoom window
#
# Option A:
#   Leave zoom_bounds = None for an automatic central zoom window.
#
# Option B:
#   Manually set zoom_bounds = (xmin, xmax, ymin, ymax)
#   after inspecting the first output.
# ---------------------------------------------------------------------

zoom_bounds = None

# Example manual override (replace with your own values if wanted):
# zoom_bounds = (
#     360800, 361800,   # xmin, xmax
#     6272450, 6273450  # ymin, ymax
# )

# ---------------------------------------------------------------------
# Get full shoreline extent
# ---------------------------------------------------------------------

xmin, ymin, xmax, ymax = shorelines_plot.total_bounds

print("Full shoreline bounds:")
print(f"  xmin={xmin:.1f}, xmax={xmax:.1f}")
print(f"  ymin={ymin:.1f}, ymax={ymax:.1f}")

# ---------------------------------------------------------------------
# If not provided, create an automatic zoom window centred on the
# middle of the shoreline extent.
# ---------------------------------------------------------------------

if zoom_bounds is None:

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    xwidth = (xmax - xmin) * 0.22
    yheight = (ymax - ymin) * 0.22

    zxmin = xmid - 0.5 * xwidth
    zxmax = xmid + 0.5 * xwidth
    zymin = ymid - 0.5 * yheight
    zymax = ymid + 0.5 * yheight

else:

    zxmin, zxmax, zymin, zymax = zoom_bounds

print("Zoom bounds:")
print(f"  xmin={zxmin:.1f}, xmax={zxmax:.1f}")
print(f"  ymin={zymin:.1f}, ymax={zymax:.1f}")

# ---------------------------------------------------------------------
# Background image: 2020 MNDWI
# ---------------------------------------------------------------------

background_year = 2020
background = yearly_ds[water_index].sel(year=background_year)

# ---------------------------------------------------------------------
# Colour ramp for all years
# ---------------------------------------------------------------------

years = sorted(shorelines_plot["year"].unique())

norm = Normalize(
    vmin=min(years),
    vmax=max(years),
)

cmap = cm.get_cmap("magma")

# ---------------------------------------------------------------------
# Make figure
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 8),
)

ax0 = axes[0]
ax1 = axes[1]

# ---------------------------------------------------------------------
# PANEL 1: Full context
# ---------------------------------------------------------------------

background.plot(
    ax=ax0,
    cmap="Greys",
    vmin=-1,
    vmax=1,
    alpha=0.45,
    add_colorbar=False,
)

for year in years:

    gdf_year = shorelines_plot[
        shorelines_plot["year"] == year
    ]

    gdf_year.plot(
        ax=ax0,
        color=cmap(norm(year)),
        linewidth=1.5,
    )

# Show zoom box on context panel
zoom_rect = Rectangle(
    (zxmin, zymin),
    zxmax - zxmin,
    zymax - zymin,
    fill=False,
    linewidth=2,
    linestyle="--",
)
ax0.add_patch(zoom_rect)

ax0.set_title(
    "Montrose annual shorelines (full context)"
)

ax0.set_aspect("equal")

# ---------------------------------------------------------------------
# PANEL 2: Zoomed section
# ---------------------------------------------------------------------

background.plot(
    ax=ax1,
    cmap="Greys",
    vmin=-1,
    vmax=1,
    alpha=0.45,
    add_colorbar=False,
)

for year in years:

    gdf_year = shorelines_plot[
        shorelines_plot["year"] == year
    ]

    gdf_year.plot(
        ax=ax1,
        color=cmap(norm(year)),
        linewidth=2.0,
    )

ax1.set_xlim(zxmin, zxmax)
ax1.set_ylim(zymin, zymax)

ax1.set_title(
    "Montrose annual shorelines (zoomed section)"
)

ax1.set_aspect("equal")

# ---------------------------------------------------------------------
# Shared formatting
# ---------------------------------------------------------------------

for ax in axes:
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")

# ---------------------------------------------------------------------
# Colorbar for year
# ---------------------------------------------------------------------

sm = cm.ScalarMappable(
    norm=norm,
    cmap=cmap,
)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    fraction=0.03,
    pad=0.02,
)

cbar.set_label("Year")

plt.tight_layout()

qa_file = (
    qa_dir
    / "montrose_shorelines_QA_magma_zoom.png"
)

plt.savefig(
    qa_file,
    dpi=250,
    bbox_inches="tight",
)

plt.close()

print()
print(f"Saved improved QA plot: {qa_file}")


# =====================================================================
# FINAL CHECKS
# =====================================================================

expected_years = set(
    range(
        start_year,
        end_year + 1,
    )
)


actual_years = set(
    contours_gdf.index.unique()
)


missing_years = sorted(
    expected_years
    - actual_years
)


print()
print("=" * 70)
print("FINISHED")
print("=" * 70)


if missing_years:

    print(
        "Years with no extracted shoreline:"
    )

    print(
        missing_years
    )

else:

    print(
        "Shorelines extracted for all years."
    )