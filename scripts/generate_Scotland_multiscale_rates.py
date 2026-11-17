#!/usr/bin/env python
"""Create multi-scale Scotland DEA Coastlines rate layers for QGIS/CMV testing.

Inputs
------
1. National merged annual shoreline GeoPackage (BNG).
2. National merged 30 m rates-of-change GeoPackage (BNG).

Outputs
-------
One GeoPackage containing:
    rates_5000m
    rates_1000m
    rates_300m
    rates_30m

For aggregated layers:
    - generate regularly spaced points along the 2024 baseline shoreline;
    - buffer each point to create a moving coastal window;
    - select good-quality 30 m DEA rate points within each window;
    - take the median dist_YYYY shoreline position for every year;
    - regress the median annual positions to calculate rates.

Scotland temporal rates:
    1988-2025
    2000-2025
    2015-2025

For this first QGIS test, aggregation radius is set equal to output point
spacing. Change SCALES if you want different radii later.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from coastlines.vector import change_regress, points_on_line


# =============================================================================
# SETTINGS
# =============================================================================

HOME_FOLDER = Path("/media/nevis_array/user_homes/mhurst/")
RESULTS_FOLDER = HOME_FOLDER / "scotland_dea_results"

CRS = "EPSG:27700"
BASELINE_YEAR = 2024

VECTOR_VERSION = "scotland_grid_v0.2_closing5"

INPUT_DIR = (
    RESULTS_FOLDER
    / "dea_analysis"
    / VECTOR_VERSION
    / "Scotland"
)

SHORELINES_FILE = (
    INPUT_DIR
    / "Scotland_DEA_MSL_annual_shorelines_1988_2025_BNG.gpkg"
)

RATES_FILE = (
    INPUT_DIR
    / f"Scotland_DEA_MSL_ratesofchange_baseline{BASELINE_YEAR}_BNG.gpkg"
)

OUTPUT_GPKG = (
    INPUT_DIR
    / f"Scotland_DEA_MSL_multiscale_rates_baseline{BASELINE_YEAR}_BNG.gpkg"
)

# Output spacing and moving-window radius.
# 30 m is kept as the original unaggregated layer.
SCALES = {
    "5000m": {"spacing_m": 2500, "radius_m": 5000},
    "1000m": {"spacing_m": 500, "radius_m": 1000},
    "300m": {"spacing_m": 150, "radius_m": 300},
}

RATE_PERIODS = {
    "1988_2025": (1988, 2025),
    "2000_2025": (2000, 2025),
    "2015_2025": (2015, 2025),
}

GOOD_ONLY = True
MIN_SOURCE_POINTS = 3
MIN_REGRESSION_OBS = 3


# =============================================================================
# RATE HELPERS
# =============================================================================

def period_regression(row, start_year, end_year, suffix):
    """Run DEA change_regress() for one explicit time period."""

    years = np.arange(start_year, end_year + 1)

    dist_cols = [
        f"dist_{year}"
        for year in years
        if f"dist_{year}" in row.index
    ]

    years = np.array(
        [int(column.replace("dist_", "")) for column in dist_cols],
        dtype=int,
    )

    slope_field = f"rate_{suffix}"
    intercept_field = f"incpt_{suffix}"
    pvalue_field = f"sig_{suffix}"
    stderr_field = f"se_{suffix}"
    outliers_field = f"outl_{suffix}"

    empty_result = pd.Series(
        {
            slope_field: np.nan,
            intercept_field: np.nan,
            pvalue_field: np.nan,
            stderr_field: np.nan,
            outliers_field: " ".join(map(str, years)),
        }
    )

    if not dist_cols:
        return empty_result

    values = row[dist_cols].to_numpy(dtype=float)

    if np.isfinite(values).sum() < MIN_REGRESSION_OBS:
        return empty_result

    try:
        return change_regress(
            y_vals=values,
            x_vals=years,
            x_labels=years,
            slope_var=slope_field,
            interc_var=intercept_field,
            pvalue_var=pvalue_field,
            stderr_var=stderr_field,
            outliers_var=outliers_field,
        )
    except Exception:
        return empty_result


def add_period_rates(gdf):
    """Calculate all configured temporal rates from dist_YYYY fields."""

    gdf = gdf.copy()

    for suffix, (start_year, end_year) in RATE_PERIODS.items():
        print(f"    Calculating rate {start_year}-{end_year}...")

        result = gdf.apply(
            lambda row: period_regression(
                row,
                start_year=start_year,
                end_year=end_year,
                suffix=suffix,
            ),
            axis=1,
        )

        rate_fields = [
            f"rate_{suffix}",
            f"incpt_{suffix}",
            f"sig_{suffix}",
            f"se_{suffix}",
            f"outl_{suffix}",
        ]

        gdf[rate_fields] = result

        dist_cols = [
            f"dist_{year}"
            for year in range(start_year, end_year + 1)
            if f"dist_{year}" in gdf.columns
        ]

        years = [
            int(column.replace("dist_", ""))
            for column in dist_cols
        ]

        outlier_field = f"outl_{suffix}"
        valid_field = f"valid_obs_{suffix}"

        def count_valid(row):
            excluded = set(str(row[outlier_field]).split())
            return sum(
                pd.notna(row[column])
                and str(year) not in excluded
                for column, year in zip(dist_cols, years)
            )

        gdf[valid_field] = gdf.apply(count_valid, axis=1)

    return gdf


# =============================================================================
# HOTSPOT GENERATION
# =============================================================================

def prepare_baseline(shorelines):
    """Return the baseline shoreline in the form expected by points_on_line()."""

    baseline = shorelines.loc[
        shorelines["year"].astype(int) == BASELINE_YEAR,
        ["geometry"],
    ].copy()

    if baseline.empty:
        raise ValueError(
            f"No {BASELINE_YEAR} shoreline found in {SHORELINES_FILE}"
        )

    baseline["year"] = str(BASELINE_YEAR)
    baseline = baseline.set_index("year")

    return baseline


def prepare_source_rates(rates):
    """Keep source geometry and annual distance fields used by aggregation."""

    source = rates.copy()

    if GOOD_ONLY and "certainty" in source.columns:
        before = len(source)
        source = source.loc[source["certainty"] == "good"].copy()
        print(
            f"Using {len(source):,} / {before:,} raw points "
            "with certainty='good'"
        )

    dist_fields = sorted(
        [
            column
            for column in source.columns
            if column.startswith("dist_")
        ],
        key=lambda name: int(name.split("_", 1)[1]),
    )

    if not dist_fields:
        raise ValueError(
            "No dist_YYYY fields found in the national rates layer."
        )

    return source[dist_fields + ["geometry"]].copy(), dist_fields


def generate_aggregate_layer(
    baseline,
    source_rates,
    dist_fields,
    label,
    spacing_m,
    radius_m,
):
    """Generate one DEA-style moving-window rates layer."""

    print()
    print("=" * 72)
    print(f"Generating {label} layer")
    print(f"  Point spacing: {spacing_m:,} m")
    print(f"  Window radius: {radius_m:,} m")
    print("=" * 72)

    hotspots = points_on_line(
        baseline,
        index=str(BASELINE_YEAR),
        distance=int(spacing_m),
    ).reset_index(drop=True)

    hotspots["hotspot_id"] = np.arange(1, len(hotspots) + 1)

    print(f"  Candidate hotspot points: {len(hotspots):,}")

    windows = hotspots[["hotspot_id", "geometry"]].copy()
    windows["geometry"] = windows.geometry.buffer(radius_m)

    joined = gpd.sjoin(
        source_rates,
        windows,
        how="inner",
        predicate="within",
    )

    if joined.empty:
        raise RuntimeError(
            f"No source rate points joined to {label} hotspot windows."
        )

    grouped = joined.groupby("index_right")

    medians = grouped[dist_fields].median().round(2)
    counts = grouped.size().rename("n_source_points")
    summary = medians.join(counts)

    hotspots = hotspots.join(
        summary,
        how="left",
    )

    hotspots["n_source_points"] = (
        hotspots["n_source_points"]
        .fillna(0)
        .astype(int)
    )

    before = len(hotspots)

    hotspots = hotspots.loc[
        hotspots["n_source_points"] >= MIN_SOURCE_POINTS
    ].copy()

    print(
        f"  Retained hotspot points: {len(hotspots):,} "
        f"(dropped {before - len(hotspots):,} with fewer than "
        f"{MIN_SOURCE_POINTS} source points)"
    )

    hotspots = add_period_rates(hotspots)

    hotspots["scale_m"] = radius_m
    hotspots["spacing_m"] = spacing_m
    hotspots["radius_m"] = radius_m
    hotspots["baseline_year"] = BASELINE_YEAR
    hotspots["indicator"] = "DEA_MSL"
    hotspots["aggregation"] = "median annual shoreline position"

    headline_fields = [
        "hotspot_id",
        "scale_m",
        "spacing_m",
        "radius_m",
        "n_source_points",
        "indicator",
        "baseline_year",
        "aggregation",
    ]

    for suffix in RATE_PERIODS:
        headline_fields.extend(
            [
                f"rate_{suffix}",
                f"sig_{suffix}",
                f"se_{suffix}",
                f"outl_{suffix}",
                f"valid_obs_{suffix}",
            ]
        )

    ordered = (
        [
            column
            for column in headline_fields
            if column in hotspots.columns
        ]
        + dist_fields
        + ["geometry"]
    )

    return hotspots[ordered]


def prepare_raw_30m(rates):
    """Prepare original DEA 30 m point layer for the same GeoPackage."""

    raw = rates.copy()

    raw["scale_m"] = 30
    raw["radius_m"] = 0
    raw["n_source_points"] = 1
    raw["aggregation"] = "none - original DEA rate point"

    return raw


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("=" * 72)
    print("SCOTLAND DEA COASTLINES — MULTI-SCALE RATE LAYERS")
    print("=" * 72)
    print(f"Shorelines: {SHORELINES_FILE}")
    print(f"Rates:      {RATES_FILE}")
    print(f"Output:     {OUTPUT_GPKG}")
    print("=" * 72)

    if not SHORELINES_FILE.exists():
        raise FileNotFoundError(SHORELINES_FILE)

    if not RATES_FILE.exists():
        raise FileNotFoundError(RATES_FILE)

    shorelines = gpd.read_file(
        SHORELINES_FILE,
        layer="annual_shorelines",
    ).to_crs(CRS)

    rates = gpd.read_file(
        RATES_FILE,
        layer="rates_of_change",
    ).to_crs(CRS)

    print(f"Loaded {len(shorelines):,} annual shoreline features")
    print(f"Loaded {len(rates):,} raw 30 m rate points")

    baseline = prepare_baseline(shorelines)
    source_rates, dist_fields = prepare_source_rates(rates)

    if OUTPUT_GPKG.exists():
        print(f"Removing previous output: {OUTPUT_GPKG}")
        OUTPUT_GPKG.unlink()

    for label, settings in SCALES.items():
        layer = generate_aggregate_layer(
            baseline=baseline,
            source_rates=source_rates,
            dist_fields=dist_fields,
            label=label,
            spacing_m=settings["spacing_m"],
            radius_m=settings["radius_m"],
        )

        layer_name = f"rates_{label}"

        layer.to_file(
            OUTPUT_GPKG,
            layer=layer_name,
            driver="GPKG",
        )

        print(
            f"  Wrote {layer_name}: {len(layer):,} points"
        )

    raw = prepare_raw_30m(rates)

    raw.to_file(
        OUTPUT_GPKG,
        layer="rates_30m",
        driver="GPKG",
    )

    print()
    print(f"Wrote rates_30m: {len(raw):,} points")

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(OUTPUT_GPKG)
    print()
    print("Layers:")
    print("  rates_5000m")
    print("  rates_1000m")
    print("  rates_300m")
    print("  rates_30m")


if __name__ == "__main__":
    main()
