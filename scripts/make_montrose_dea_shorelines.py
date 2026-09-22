from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from dea_tools.spatial import subpixel_contours

from coastlines.vector import (
    load_rasters,
    contours_preprocess,
    contour_certainty,
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

tide_model = "EOT20"

output_dir = Path(
    "outputs/montrose_vectors"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# LOAD DEA-STYLE ANNUAL RASTERS
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
# DEA Africa uses known tidal-model/ocean points to identify which
# connected water body is the ocean.
#
# For the Montrose prototype the eastern edge of the AOI lies offshore
# in the North Sea, so points along this edge provide equivalent seeds.
# =====================================================================

x_values = yearly_ds.x.values
y_values = yearly_ds.y.values


# A few pixels inside the eastern edge
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
    f"North Sea ocean seed points."
)


# =====================================================================
# DEA AFRICA PREPROCESSING
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

    # ESA WorldCover is not currently available through our local
    # ODC setup. NDWI and temporal filtering remain enabled.
    mask_landcover=False,

    mask_ndwi=True,
    mask_temporal=True,

    mask_modifications=None,
)


print(
    "Preprocessing complete."
)


# =====================================================================
# EXTRACT SUBPIXEL SHORELINES
#
# This is the native DEA Africa shoreline extraction step.
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
).set_index(
    "year"
)


print(
    f"Extracted {len(contours_gdf)} "
    f"raw shoreline features."
)


# =====================================================================
# APPLY DEA CERTAINTY ATTRIBUTION
#
# certainty_masks uses string year keys in the original DEA Africa
# implementation. Normalising both sides here avoids int/string
# mismatches between library versions.
# =====================================================================

print()
print("=" * 70)
print("APPLYING CERTAINTY ATTRIBUTION")
print("=" * 70)


contours_gdf.index = (
    contours_gdf.index
    .astype(str)
)


certainty_masks = {
    str(year): mask
    for year, mask
    in certainty_masks.items()
}


contours_gdf = contour_certainty(
    contours_gdf,
    certainty_masks,
)


# Convert year back to integer after DEA certainty attribution.
contours_gdf.index = (
    contours_gdf.index
    .astype(int)
)

contours_gdf.index.name = "year"


# Remove empty geometries defensively.
contours_gdf = contours_gdf[
    contours_gdf.geometry.notnull()
    & ~contours_gdf.geometry.is_empty
].copy()


print(
    f"Certainty attribution produced "
    f"{len(contours_gdf)} shoreline segments."
)


# =====================================================================
# ADD SCOTLAND / CMT PROVENANCE
# =====================================================================

contours_gdf["indicator"] = "DEA_MSL"

contours_gdf["water_index"] = water_index

contours_gdf["threshold"] = float(
    index_threshold
)

contours_gdf["tide_model"] = tide_model

contours_gdf["tide_datum"] = (
    "model MSL"
)

contours_gdf["method"] = (
    "DE Africa Coastlines subpixel contour"
)

contours_gdf["raster_version"] = (
    raster_version
)


# =====================================================================
# CREATE UNIQUE SEGMENT IDs
#
# contour_certainty() can split a single annual shoreline into several
# segments where certainty classifications change.
# =====================================================================

shorelines = (
    contours_gdf
    .reset_index()
)


shorelines["segment_no"] = (
    shorelines
    .groupby("year")
    .cumcount()
    + 1
)


shorelines["shoreline_id"] = (
    shorelines["year"].astype(str)
    + "_"
    + shorelines["segment_no"]
    .astype(str)
    .str.zfill(3)
)


# Put useful fields first.
field_order = [
    "shoreline_id",
    "year",
    "segment_no",
    "indicator",
    "certainty",
    "water_index",
    "threshold",
    "tide_model",
    "tide_datum",
    "method",
    "raster_version",
    "geometry",
]


shorelines = shorelines[
    [
        column
        for column in field_order
        if column in shorelines.columns
    ]
]


# =====================================================================
# QA SUMMARY
#
# Calculate lengths BEFORE converting to EPSG:4326.
# =====================================================================

shorelines["length_m"] = (
    shorelines.geometry.length
)


summary = (
    shorelines
    .groupby(
        [
            "year",
            "certainty",
        ],
        dropna=False,
    )
    .agg(
        n_segments=(
            "shoreline_id",
            "count",
        ),
        length_m=(
            "length_m",
            "sum",
        ),
    )
    .reset_index()
)


summary["length_m"] = (
    summary["length_m"]
    .round(1)
)


summary_file = (
    output_dir
    / "montrose_shoreline_certainty_summary.csv"
)


summary.to_csv(
    summary_file,
    index=False,
)


print()
print(
    "Certainty classes:"
)

print(
    shorelines["certainty"]
    .value_counts(
        dropna=False
    )
)


print()
print(
    f"Saved certainty summary: "
    f"{summary_file}"
)


# =====================================================================
# REMOVE TEMPORARY LENGTH ATTRIBUTE
# =====================================================================

shorelines = shorelines.drop(
    columns="length_m"
)


# =====================================================================
# EXPORT CMT HANDOFF PRODUCT
#
# GeoJSON is written in EPSG:4326.
# =====================================================================

output_geojson = (
    output_dir
    / "montrose_DEA_MSL_annual_shorelines_1988_2025.geojson"
)


shorelines_wgs84 = (
    shorelines
    .to_crs(
        "EPSG:4326"
    )
)


shorelines_wgs84.to_file(
    output_geojson,
    driver="GeoJSON",
)


print()
print("=" * 70)
print("OUTPUT COMPLETE")
print("=" * 70)

print()
print(
    f"GeoJSON: {output_geojson}"
)

print(
    f"Number of shoreline segments: "
    f"{len(shorelines_wgs84)}"
)

print(
    f"Years: "
    f"{shorelines_wgs84.year.min()}-"
    f"{shorelines_wgs84.year.max()}"
)

print()


missing_years = sorted(
    set(
        range(
            start_year,
            end_year + 1,
        )
    )
    - set(
        shorelines_wgs84.year.unique()
    )
)


if missing_years:

    print(
        f"WARNING: no shoreline extracted "
        f"for years: {missing_years}"
    )

else:

    print(
        "Shorelines present for every year."
    )


print()
print("Done.")