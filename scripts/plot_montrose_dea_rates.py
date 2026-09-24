from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator

from coastlines.vector import load_rasters


# =====================================================================
# SETTINGS
# =====================================================================

rates_file = Path(
    "outputs/montrose_tide_QA/"
    "montrose_DEA_rates_30m.geojson"
)

shoreline_file = Path(
    "outputs/montrose_vectors/"
    "montrose_DEA_MSL_annual_shorelines_1988_2025.geojson"
)

output_dir = Path(
    "outputs/montrose_tide_QA"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

output_png = (
    output_dir
    / "montrose_DEA_rate_map.png"
)


# Raster settings
raster_path = "outputs/dea_rasters"
raster_version = "scotland_v0.1"
study_area = "montrose"

start_year = 1988
end_year = 2025

water_index = "mndwi"

background_year = 2020


# Amount of padding around rate points, metres
map_padding = 75


# =====================================================================
# LOAD VECTORS
# =====================================================================

print()
print("=" * 70)
print("LOADING VECTOR DATA")
print("=" * 70)


rates = gpd.read_file(
    rates_file
)

shorelines = gpd.read_file(
    shoreline_file
)


candidate_columns = [
    "rate_time",
    "rate",
    "slope",
    "rate_ols",
    "change_rate",
]


rate_col = next(
    (
        col
        for col in candidate_columns
        if col in rates.columns
    ),
    None,
)


if rate_col is None:
    raise ValueError(
        "Could not identify rate column. "
        f"Available columns: {list(rates.columns)}"
    )


print(
    f"Using rate column: {rate_col}"
)


# =====================================================================
# LOAD BACKGROUND
# =====================================================================

print()
print("=" * 70)
print("LOADING BACKGROUND RASTER")
print("=" * 70)


yearly_ds, _ = load_rasters(
    path=raster_path,
    raster_version=raster_version,
    study_area=study_area,
    water_index=water_index,
    start_year=start_year,
    end_year=end_year,
)


raster_crs = yearly_ds.rio.crs


if raster_crs is None:
    raise ValueError(
        "Raster dataset has no CRS."
    )


background = (
    yearly_ds[water_index]
    .sel(year=background_year)
)


# =====================================================================
# COMMON CRS
# =====================================================================

rates = rates.to_crs(
    raster_crs
)

shorelines = shorelines.to_crs(
    raster_crs
)


shorelines["year"] = (
    shorelines["year"]
    .astype(int)
)


baseline = (
    shorelines[
        shorelines["year"] == background_year
    ]
    .dissolve()
)


# =====================================================================
# CLEAN RATE DATA
# =====================================================================

rates = rates.copy()


rates[rate_col] = (
    rates[rate_col]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
)


rates = rates.dropna(
    subset=[rate_col]
)


print()
print(
    rates[rate_col].describe()
)


# =====================================================================
# COLOUR SCALE
# =====================================================================

colour_limit = np.nanpercentile(
    np.abs(
        rates[rate_col]
    ),
    95,
)


if colour_limit == 0:
    colour_limit = 1.0


norm = TwoSlopeNorm(
    vmin=-colour_limit,
    vcenter=0,
    vmax=colour_limit,
)


# =====================================================================
# DEFINE MAP AREA
#
# Create requested bounds from the points, but constrain them to the
# actual raster extent.
# =====================================================================

pxmin, pymin, pxmax, pymax = (
    rates.total_bounds
)


requested_xmin = pxmin - map_padding
requested_xmax = pxmax + map_padding

requested_ymin = pymin - map_padding
requested_ymax = pymax + map_padding


# Actual raster bounds
rxmin, rymin, rxmax, rymax = (
    background.rio.bounds()
)


crop_xmin = max(
    requested_xmin,
    rxmin,
)

crop_xmax = min(
    requested_xmax,
    rxmax,
)

crop_ymin = max(
    requested_ymin,
    rymin,
)

crop_ymax = min(
    requested_ymax,
    rymax,
)


# =====================================================================
# CROP RASTER
#
# The map axis limits will subsequently come FROM THIS RASTER.
# This guarantees no white gap between raster and axes.
# =====================================================================

background_crop = background.rio.clip_box(
    minx=crop_xmin,
    miny=crop_ymin,
    maxx=crop_xmax,
    maxy=crop_ymax,
)


# Pixel-edge bounds of the cropped raster
map_xmin, map_ymin, map_xmax, map_ymax = (
    background_crop.rio.bounds()
)


print()
print("Map extent:")
print(
    map_xmin,
    map_xmax,
    map_ymin,
    map_ymax,
)


# =====================================================================
# CREATE FIGURE
#
# Dedicated colourbar column prevents labels colliding with the map.
# =====================================================================

print()
print("=" * 70)
print("CREATING MAP")
print("=" * 70)


fig = plt.figure(
    figsize=(7.2, 9.2)
)


gs = fig.add_gridspec(
    nrows=1,
    ncols=2,
    width_ratios=[
        1.0,
        0.055,
    ],
    left=0.11,
    right=0.88,
    bottom=0.09,
    top=0.93,
    wspace=0.09,
)


ax = fig.add_subplot(
    gs[0]
)

cax = fig.add_subplot(
    gs[1]
)


# =====================================================================
# BACKGROUND
# =====================================================================

background_crop.plot(
    ax=ax,
    cmap="Greys",
    vmin=-1,
    vmax=1,
    alpha=0.38,
    add_colorbar=False,
    zorder=1,
)


# =====================================================================
# BASELINE SHORELINE
# =====================================================================

baseline.plot(
    ax=ax,
    color="0.25",
    linewidth=0.8,
    alpha=0.7,
    zorder=2,
)


# =====================================================================
# RATE POINTS
#
# Negative = erosion = red
# Positive = accretion = blue
# =====================================================================

scatter = ax.scatter(
    rates.geometry.x,
    rates.geometry.y,
    c=rates[rate_col],
    cmap="coolwarm_r",
    norm=norm,
    s=22,
    edgecolors="0.25",
    linewidths=0.20,
    zorder=3,
)


# =====================================================================
# EXACTLY MATCH AXES TO RASTER
# =====================================================================

ax.set_xlim(
    map_xmin,
    map_xmax,
)

ax.set_ylim(
    map_ymin,
    map_ymax,
)


ax.set_aspect(
    "equal",
    adjustable="box",
)


# =====================================================================
# AXES
# =====================================================================

ax.set_xlabel(
    "Easting (km)",
    fontsize=11,
)

ax.set_ylabel(
    "Northing (km)",
    fontsize=11,
)


ax.xaxis.set_major_locator(
    MaxNLocator(
        nbins=4,
    )
)

ax.yaxis.set_major_locator(
    MaxNLocator(
        nbins=6,
    )
)


ax.xaxis.set_major_formatter(
    FuncFormatter(
        lambda value, pos:
        f"{value / 1000:.1f}"
    )
)


ax.yaxis.set_major_formatter(
    FuncFormatter(
        lambda value, pos:
        f"{value / 1000:.1f}"
    )
)


ax.tick_params(
    axis="both",
    labelsize=9,
)


# =====================================================================
# TITLE
# =====================================================================

ax.set_title(
    "Montrose DEA shoreline change rates",
    fontsize=14,
    pad=10,
)


# =====================================================================
# COLOURBAR
# =====================================================================

cbar = fig.colorbar(
    scatter,
    cax=cax,
)


cbar.ax.tick_params(
    labelsize=9,
)


cbar.set_label(
    "Rate of shoreline change (m/year)",
    fontsize=10,
    labelpad=10,
)


# ---------------------------------------------------------------------
# Sign convention
#
# These now sit above/below the colourbar itself, not in map space.
# ---------------------------------------------------------------------

cbar.ax.text(
    0.5,
    1.025,
    "Accretion\nseaward",
    ha="center",
    va="bottom",
    transform=cbar.ax.transAxes,
    fontsize=9,
)


cbar.ax.text(
    0.5,
    -0.025,
    "Erosion\nlandward",
    ha="center",
    va="top",
    transform=cbar.ax.transAxes,
    fontsize=9,
)


# =====================================================================
# SAVE
# =====================================================================

fig.savefig(
    output_png,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print()
print(
    f"Saved map: {output_png}"
)

print()
print("Done.")
