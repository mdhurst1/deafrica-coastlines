from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from scipy.stats import linregress
from shapely.ops import nearest_points

from coastlines.vector import (
    load_rasters,
    points_on_line,
    annual_movements,
    calculate_regressions,
)


# =====================================================================
# SETTINGS
# =====================================================================

raster_path = "outputs/dea_rasters"
raster_version = "scotland_v0.1"
study_area = "montrose"

shoreline_file = Path(
    "outputs/montrose_vectors/"
    "montrose_DEA_MSL_annual_shorelines_1988_2025.geojson"
)

start_year = 1988
end_year = 2025

baseline_year = 2020

water_index = "mndwi"

point_spacing = 30

output_dir = Path(
    "outputs/montrose_tide_QA"
)

output_dir.mkdir(
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

yearly_ds, _ = load_rasters(
    path=raster_path,
    raster_version=raster_version,
    study_area=study_area,
    water_index=water_index,
    start_year=start_year,
    end_year=end_year,
)

crs = yearly_ds.rio.crs

if crs is None:
    raise ValueError("Raster stack has no CRS.")

print(f"CRS: {crs}")


# =====================================================================
# LOAD SHORELINES
# =====================================================================

print()
print("=" * 70)
print("LOADING SHORELINES")
print("=" * 70)

shorelines = gpd.read_file(
    shoreline_file
)

shorelines = shorelines.to_crs(
    crs
)

shorelines["year"] = (
    shorelines["year"]
    .astype(int)
)

print(
    f"Loaded {len(shorelines)} attributed shoreline segments."
)


# =====================================================================
# DISSOLVE CERTAINTY SEGMENTS TO ONE GEOMETRY PER YEAR
#
# This is important because DEA annual_movements() takes the first
# geometry for each annual index. Dissolving restores the complete
# annual shoreline before applying the DEA rate method.
# =====================================================================

contours_gdf = (
    shorelines[
        [
            "year",
            "geometry",
        ]
    ]
    .dissolve(
        by="year"
    )
)


# DEA functions expect year labels as strings
contours_gdf.index = (
    contours_gdf.index
    .astype(str)
)

contours_gdf.index.name = "year"


print(
    f"Dissolved to {len(contours_gdf)} annual shorelines."
)


# =====================================================================
# GENERATE DEA BASELINE POINTS
# =====================================================================

print()
print("=" * 70)
print("GENERATING DEA BASELINE POINTS")
print("=" * 70)


points_gdf = points_on_line(
    contours_gdf,
    str(baseline_year),
    distance=point_spacing,
)


# Give each point a stable ID
points_gdf["point_id"] = np.arange(
    len(points_gdf)
)

points_gdf = points_gdf.set_index(
    "point_id",
    drop=False,
)


print(
    f"Generated {len(points_gdf)} points "
    f"at {point_spacing} m spacing."
)


# Keep an untouched copy because annual_movements modifies its input
baseline_points = points_gdf.copy()


# =====================================================================
# RUN NATIVE DEA ANNUAL MOVEMENT CALCULATION
# =====================================================================

print()
print("=" * 70)
print("CALCULATING DEA ANNUAL MOVEMENTS")
print("=" * 70)


movement_gdf = annual_movements(
    points_gdf=points_gdf.copy(),
    contours_gdf=contours_gdf,
    yearly_ds=yearly_ds,
    baseline_year=str(baseline_year),
    water_index=water_index,
    max_valid_dist=5000,
)


print(
    "Annual shoreline movements calculated."
)


# =====================================================================
# DEA LONG-TERM RATE
# =====================================================================

print()
print("=" * 70)
print("CALCULATING DEA RATES")
print("=" * 70)


rates_gdf = calculate_regressions(
    movement_gdf.copy(),
    contours_gdf,
)


# Preserve the point ID as a normal attribute column,
# then remove it from the dataframe index.
rates_gdf = rates_gdf.copy()

rates_gdf["point_id"] = (
    rates_gdf.index.astype(int)
)

rates_gdf = rates_gdf.reset_index(
    drop=True
)


rates_output = (
    output_dir
    / "montrose_DEA_rates_30m.geojson"
)


rates_gdf.to_crs(
    "EPSG:4326"
).to_file(
    rates_output,
    driver="GeoJSON",
    index=False,
)


print(
    f"Saved DEA rates: {rates_output}"
)


# =====================================================================
# SAMPLE LOCAL ANNUAL TIDE AT EACH NEAREST SHORELINE POSITION
#
# We repeat DEA's nearest-point operation here solely so that we can
# sample tide_m at the actual annual comparison shoreline position.
# =====================================================================

print()
print("=" * 70)
print("SAMPLING LOCAL ANNUAL TIDE")
print("=" * 70)


records = []


for year in sorted(
    contours_gdf.index.astype(int)
):

    year_str = str(
        year
    )

    print(
        f"Processing {year}"
    )


    annual_contour = (
        contours_gdf
        .loc[[year_str]]
        .geometry
        .iloc[0]
    )


    tide_array = (
        yearly_ds["tide_m"]
        .sel(year=year)
    )


    dist_column = (
        f"dist_{year}"
    )


    for point_id, row in baseline_points.iterrows():

        baseline_point = (
            row.geometry
        )


        # Same nearest shoreline concept used by DEA
        comparison_point = nearest_points(
            baseline_point,
            annual_contour,
        )[1]


        # Get signed DEA shoreline distance
        distance = (
            movement_gdf
            .loc[
                point_id,
                dist_column,
            ]
            if dist_column in movement_gdf.columns
            else np.nan
        )


        # Sample local annual contributing tide
        tide_value = tide_array.interp(
            x=float(comparison_point.x),
            y=float(comparison_point.y),
            method="linear",
        ).values


        try:
            tide_value = float(
                tide_value
            )

        except (TypeError, ValueError):
            tide_value = np.nan


        records.append(
            {
                "point_id": point_id,
                "year": year,
                "distance_m": distance,
                "tide_m": tide_value,
                "baseline_x": baseline_point.x,
                "baseline_y": baseline_point.y,
                "shoreline_x": comparison_point.x,
                "shoreline_y": comparison_point.y,
            }
        )


qa_df = pd.DataFrame(
    records
)


qa_csv = (
    output_dir
    / "montrose_DEA_tide_QA_long.csv"
)


qa_df.to_csv(
    qa_csv,
    index=False,
)


print(
    f"Saved tide QA table: {qa_csv}"
)


# =====================================================================
# PER-POINT RESIDUAL TIDE TEST
#
# For each DEA point:
#
#   1. remove linear temporal trend from shoreline position
#   2. remove linear temporal trend from tide
#   3. regress residual shoreline position against residual tide
#
# beta_tide therefore has units:
#
#       metres shoreline movement / metre tide
#
# Under the DEA sign convention:
#   positive distance = seaward
#   negative distance = landward
#
# If higher residual tide systematically shifts the shoreline landward,
# we would generally expect beta_tide to be negative.
# =====================================================================

print()
print("=" * 70)
print("TESTING RESIDUAL TIDE EFFECT")
print("=" * 70)


results = []


for point_id, df in qa_df.groupby(
    "point_id"
):

    df = (
        df[
            [
                "year",
                "distance_m",
                "tide_m",
            ]
        ]
        .dropna()
        .copy()
    )


    if len(df) < 15:
        continue


    year = df["year"].values.astype(
        float
    )

    distance = df[
        "distance_m"
    ].values.astype(
        float
    )

    tide = df[
        "tide_m"
    ].values.astype(
        float
    )


    # -------------------------------------------------------------
    # Remove temporal trend from shoreline position
    # -------------------------------------------------------------

    distance_trend = linregress(
        year,
        distance,
    )


    distance_residual = (
        distance
        - (
            distance_trend.intercept
            + distance_trend.slope
            * year
        )
    )


    # -------------------------------------------------------------
    # Remove temporal trend from annual tide
    # -------------------------------------------------------------

    tide_trend = linregress(
        year,
        tide,
    )


    tide_residual = (
        tide
        - (
            tide_trend.intercept
            + tide_trend.slope
            * year
        )
    )


    # -------------------------------------------------------------
    # Residual tide regression
    # -------------------------------------------------------------

    if np.nanstd(tide_residual) == 0:
        continue


    tide_regression = linregress(
        tide_residual,
        distance_residual,
    )


    results.append(
        {
            "point_id": point_id,
            "n": len(df),
            "rate_m_per_year": distance_trend.slope,
            "beta_tide_m_per_m": tide_regression.slope,
            "tide_r": tide_regression.rvalue,
            "tide_r2": tide_regression.rvalue ** 2,
            "tide_p": tide_regression.pvalue,
            "tide_stderr": tide_regression.stderr,
        }
    )


tide_results = pd.DataFrame(
    results
)


tide_results_csv = (
    output_dir
    / "montrose_DEA_tide_effect_by_point.csv"
)


tide_results.to_csv(
    tide_results_csv,
    index=False,
)


print()
print(
    f"Saved tide regression results: "
    f"{tide_results_csv}"
)


# =====================================================================
# JOIN TIDE RESULTS TO SPATIAL POINTS
# =====================================================================

tide_points = (
    baseline_points
    .reset_index(drop=True)[
        [
            "point_id",
            "geometry",
        ]
    ]
    .merge(
        tide_results,
        on="point_id",
        how="left",
    )
)


tide_points = gpd.GeoDataFrame(
    tide_points,
    geometry="geometry",
    crs=crs,
)


tide_geojson = (
    output_dir
    / "montrose_DEA_tide_effect_30m.geojson"
)


tide_points.to_crs(
    "EPSG:4326"
).to_file(
    tide_geojson,
    driver="GeoJSON",
    index=False,
)


print(
    f"Saved spatial tide results: "
    f"{tide_geojson}"
)


# =====================================================================
# SITE-WIDE ANNUAL SUMMARY
# =====================================================================

annual_summary = (
    qa_df
    .groupby("year")
    .agg(
        median_distance_m=(
            "distance_m",
            "median",
        ),
        median_tide_m=(
            "tide_m",
            "median",
        ),
        n_points=(
            "distance_m",
            "count",
        ),
    )
    .reset_index()
)


summary_csv = (
    output_dir
    / "montrose_DEA_tide_QA_annual_summary.csv"
)


annual_summary.to_csv(
    summary_csv,
    index=False,
)


# =====================================================================
# DETREND SITE-WIDE MEDIAN SERIES
# =====================================================================

summary_valid = (
    annual_summary
    .dropna(
        subset=[
            "median_distance_m",
            "median_tide_m",
        ]
    )
    .copy()
)


years_arr = (
    summary_valid["year"]
    .values
    .astype(float)
)


distance_arr = (
    summary_valid["median_distance_m"]
    .values
)


tide_arr = (
    summary_valid["median_tide_m"]
    .values
)


distance_fit = linregress(
    years_arr,
    distance_arr,
)


tide_fit = linregress(
    years_arr,
    tide_arr,
)


summary_valid[
    "distance_residual_m"
] = (
    distance_arr
    - (
        distance_fit.intercept
        + distance_fit.slope
        * years_arr
    )
)


summary_valid[
    "tide_residual_m"
] = (
    tide_arr
    - (
        tide_fit.intercept
        + tide_fit.slope
        * years_arr
    )
)


site_tide_reg = linregress(
    summary_valid[
        "tide_residual_m"
    ],
    summary_valid[
        "distance_residual_m"
    ],
)


print()
print("=" * 70)
print("SITE-WIDE RESIDUAL TIDE RESULT")
print("=" * 70)

print(
    f"beta tide = "
    f"{site_tide_reg.slope:.2f} "
    f"m shoreline / m tide"
)

print(
    f"r = "
    f"{site_tide_reg.rvalue:.3f}"
)

print(
    f"R² = "
    f"{site_tide_reg.rvalue ** 2:.3f}"
)

print(
    f"p = "
    f"{site_tide_reg.pvalue:.4f}"
)


# =====================================================================
# QA FIGURE 1:
# ANNUAL SHORELINE POSITION AND TIDE
# =====================================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
)


axes[0].plot(
    annual_summary["year"],
    annual_summary["median_distance_m"],
    marker="o",
    linewidth=1.5,
)


axes[0].axhline(
    0,
    linewidth=0.8,
    linestyle="--",
)


axes[0].set_ylabel(
    "Median shoreline distance (m)"
)

axes[0].set_title(
    "Montrose DEA shoreline position"
)


axes[1].plot(
    annual_summary["year"],
    annual_summary["median_tide_m"],
    marker="o",
    linewidth=1.5,
)


axes[1].axhline(
    0,
    linewidth=0.8,
    linestyle="--",
)


axes[1].set_xlabel(
    "Year"
)

axes[1].set_ylabel(
    "Median contributing tide (m)"
)


fig.tight_layout()


timeseries_figure = (
    output_dir
    / "montrose_DEA_tide_timeseries.png"
)


fig.savefig(
    timeseries_figure,
    dpi=250,
    bbox_inches="tight",
)


plt.close(
    fig
)


# =====================================================================
# QA FIGURE 2:
# DETRENDED SHORELINE POSITION VS TIDE
# =====================================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)


ax.scatter(
    summary_valid["tide_residual_m"],
    summary_valid["distance_residual_m"],
    s=45,
)


x_fit = np.linspace(
    summary_valid["tide_residual_m"].min(),
    summary_valid["tide_residual_m"].max(),
    100,
)


y_fit = (
    site_tide_reg.intercept
    + site_tide_reg.slope
    * x_fit
)


ax.plot(
    x_fit,
    y_fit,
    linewidth=1.5,
)


ax.axhline(
    0,
    linewidth=0.8,
    linestyle="--",
)


ax.axvline(
    0,
    linewidth=0.8,
    linestyle="--",
)


ax.set_xlabel(
    "Detrended contributing tide (m)"
)

ax.set_ylabel(
    "Detrended shoreline position (m)"
)


ax.set_title(
    "Residual tide influence on DEA shoreline position"
)


annotation = (
    f"β = {site_tide_reg.slope:.2f} m/m\n"
    f"r = {site_tide_reg.rvalue:.2f}\n"
    f"R² = {site_tide_reg.rvalue ** 2:.2f}\n"
    f"p = {site_tide_reg.pvalue:.3f}"
)


ax.text(
    0.04,
    0.96,
    annotation,
    transform=ax.transAxes,
    va="top",
)


fig.tight_layout()


scatter_figure = (
    output_dir
    / "montrose_DEA_tide_residual_scatter.png"
)


fig.savefig(
    scatter_figure,
    dpi=250,
    bbox_inches="tight",
)


plt.close(
    fig
)

# =====================================================================
# QA FIGURE 3:
# SPATIAL VARIATION IN RESIDUAL TIDE SENSITIVITY
# =====================================================================

print()
print("=" * 70)
print("PLOTTING SPATIAL TIDE SENSITIVITY")
print("=" * 70)


# ---------------------------------------------------------------------
# Prepare results
# ---------------------------------------------------------------------

spatial_qa = tide_points.copy()


# point_id was generated sequentially along the DEA baseline shoreline
# at 30 m spacing, so this provides a simple alongshore distance axis.
spatial_qa["alongshore_m"] = (
    spatial_qa["point_id"]
    * point_spacing
)


# Sort explicitly
spatial_qa = spatial_qa.sort_values(
    "point_id"
)


# ---------------------------------------------------------------------
# Rolling median
#
# 5 points = approximately 150 m alongshore.
# This is only a visual aid; raw 30 m values remain plotted.
# ---------------------------------------------------------------------

spatial_qa["beta_rolling"] = (
    spatial_qa["beta_tide_m_per_m"]
    .rolling(
        window=5,
        center=True,
        min_periods=3,
    )
    .median()
)


# ---------------------------------------------------------------------
# Robust colour limits
#
# A small number of extreme regressions can otherwise dominate the
# colour ramp. Use symmetric 95th-percentile absolute limits.
# ---------------------------------------------------------------------

valid_beta = (
    spatial_qa["beta_tide_m_per_m"]
    .replace(
        [np.inf, -np.inf],
        np.nan,
    )
    .dropna()
)


if len(valid_beta) == 0:

    raise ValueError(
        "No valid beta_tide_m_per_m values available."
    )


colour_limit = np.nanpercentile(
    np.abs(valid_beta),
    95,
)


# Prevent pathological zero range
if colour_limit == 0:

    colour_limit = 1.0


from matplotlib.colors import TwoSlopeNorm


beta_norm = TwoSlopeNorm(
    vmin=-colour_limit,
    vcenter=0,
    vmax=colour_limit,
)


beta_cmap = plt.colormaps["coolwarm"]


# =====================================================================
# FIGURE
# =====================================================================

fig = plt.figure(
    figsize=(13, 6.5),
)


gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[
        1.45,
        1.0,
    ],
    wspace=0.18,
)


ax_profile = fig.add_subplot(
    gs[0]
)

ax_map = fig.add_subplot(
    gs[1]
)


# =====================================================================
# PANEL 1:
# ALONGSHORE TIDE SENSITIVITY
# =====================================================================

scatter = ax_profile.scatter(
    spatial_qa["alongshore_m"],
    spatial_qa["beta_tide_m_per_m"],
    c=spatial_qa["beta_tide_m_per_m"],
    cmap=beta_cmap,
    norm=beta_norm,
    s=24,
    alpha=0.80,
    zorder=3,
)


# Rolling median
ax_profile.plot(
    spatial_qa["alongshore_m"],
    spatial_qa["beta_rolling"],
    linewidth=2.0,
    color="black",
    label="150 m rolling median",
    zorder=4,
)


# Zero line
ax_profile.axhline(
    0,
    linewidth=1.0,
    linestyle="--",
    color="0.4",
    zorder=1,
)


ax_profile.set_xlabel(
    "Distance along baseline shoreline (m)"
)

ax_profile.set_ylabel(
    "Residual tide sensitivity\n"
    "(m shoreline / m tide)"
)

ax_profile.set_title(
    "Residual tide sensitivity along Montrose"
)


ax_profile.legend(
    loc="best",
    frameon=True,
)


ax_profile.grid(
    alpha=0.15
)


# =====================================================================
# PANEL 2:
# SPATIAL MAP OF TIDE SENSITIVITY
# =====================================================================

# Plot dissolved 2020 baseline shoreline for geographic context
baseline_geometry = (
    contours_gdf
    .loc[[str(baseline_year)]]
    .copy()
)


baseline_geometry.plot(
    ax=ax_map,
    color="0.35",
    linewidth=1.0,
    zorder=1,
)


# Plot points coloured by beta
map_scatter = ax_map.scatter(
    spatial_qa.geometry.x,
    spatial_qa.geometry.y,
    c=spatial_qa["beta_tide_m_per_m"],
    cmap=beta_cmap,
    norm=beta_norm,
    s=24,
    edgecolors="none",
    zorder=3,
)


ax_map.set_title(
    "Spatial pattern"
)

ax_map.set_xlabel(
    "Easting (m)"
)

ax_map.set_ylabel(
    "Northing (m)"
)

ax_map.set_aspect(
    "equal"
)


# Avoid scientific/offset notation on projected coordinates
from matplotlib.ticker import ScalarFormatter


xformatter = ScalarFormatter(
    useOffset=False
)

xformatter.set_scientific(
    False
)


yformatter = ScalarFormatter(
    useOffset=False
)

yformatter.set_scientific(
    False
)


ax_map.xaxis.set_major_formatter(
    xformatter
)

ax_map.yaxis.set_major_formatter(
    yformatter
)


# =====================================================================
# SHARED COLOURBAR
# =====================================================================

cbar = fig.colorbar(
    map_scatter,
    ax=[
        ax_profile,
        ax_map,
    ],
    fraction=0.025,
    pad=0.025,
)


cbar.set_label(
    "Residual tide sensitivity "
    "(m shoreline / m tide)"
)


# =====================================================================
# SAVE
# =====================================================================

spatial_figure = (
    output_dir
    / "montrose_DEA_tide_effect_alongshore.png"
)


fig.savefig(
    spatial_figure,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print()
print(
    f"Spatial tide QA: {spatial_figure}"
)

# =====================================================================
# SIMPLE SPATIAL SUMMARY
# =====================================================================

beta = spatial_qa[
    "beta_tide_m_per_m"
].dropna()


print()
print("=" * 70)
print("SPATIAL TIDE SENSITIVITY SUMMARY")
print("=" * 70)

print(
    f"Median beta: "
    f"{beta.median():.2f} m/m"
)

print(
    f"10th-90th percentile: "
    f"{beta.quantile(0.10):.2f} to "
    f"{beta.quantile(0.90):.2f} m/m"
)

print(
    f"Proportion negative: "
    f"{(beta < 0).mean() * 100:.1f}%"
)

print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print()
print(f"Timeseries QA: {timeseries_figure}")
print(f"Residual QA:   {scatter_figure}")
print(f"Tide results:  {tide_geojson}")

print()