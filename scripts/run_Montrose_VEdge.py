#!/usr/bin/env python3
"""Compare annual NDVI vegetation-edge candidates with and without a tidal filter."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rioxarray  # registers the .rio accessor
import xarray as xr
from dea_tools.spatial import subpixel_contours
from eo_tides.eo import pixel_tides
from shapely.ops import unary_union

from coastlines.stac import load_dem_stac, load_sentinel2_indices_stac, load_water_index_stac

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tests.vedge_diagnostics import save_diagnostics, summarise_observations

CRS = "EPSG:27700"
RESOLUTION = 30
TIDE_MODEL = "FES2022_extrapolated"
TIDE_MODEL_DIR = Path("/media/nevis_array/user_homes/mhurst/tide_models")
INPUTS = Path(__file__).resolve().parents[1] / "pilot_inputs" / "montrose_vedge"
TIDE_MODES = {"all_clear": "All clear observations", "tide_filtered": "Tide-filtered"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", type=Path, default=INPUTS / "aoi.geojson")
    parser.add_argument("--shorelines", type=Path, default=INPUTS / "msl.geojson",
                        help="MSL lines used to define the coastal search corridor.")
    parser.add_argument("--references", type=Path, default=INPUTS / "references.geojson")
    parser.add_argument("--transects", type=Path, default=INPUTS / "transects.geojson")
    parser.add_argument("--sensor", choices=("landsat", "sentinel2"), default="landsat")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tide-model-dir", type=Path, default=TIDE_MODEL_DIR)
    parser.add_argument("--tide-cache", type=Path,
                        help="Optional existing folder of {year}_tides.nc files from the AOI run.")
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2024])
    parser.add_argument("--months", type=int, nargs=2, default=[5, 9],
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--min-tide", type=float, default=0.0,
                        help="Minimum tide height in metres relative to model MSL.")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="Trial fixed NDVI contour level.")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--corridor", type=float, default=300,
                        help="Search distance in metres either side of the MSL lines.")
    parser.add_argument("--min-length", type=float, default=90,
                        help="Minimum length in metres of a retained contour segment.")
    args = parser.parse_args()
    if args.output is None:
        suffix = "_sentinel2" if args.sensor == "sentinel2" else ""
        args.output = Path(f"outputs/montrose_vedge{suffix}")
    if args.sensor == "sentinel2" and min(args.years) < 2017:
        parser.error("the Sentinel-2 pilot expects years from 2017 onwards")
    if not 1 <= args.months[0] <= args.months[1] <= 12:
        parser.error("--months must be in ascending order between 1 and 12")
    if not -1 < args.threshold < 1 or not np.isfinite(args.min_tide):
        parser.error("provide a threshold between -1 and 1 and a finite tide cutoff")
    if args.min_count < 1 or not 0 < args.corridor < np.inf or not 0 < args.min_length < np.inf:
        parser.error("minimum count, corridor distance and minimum length must be positive")
    return args


def read_bng(path):
    return gpd.read_file(path).to_crs(CRS)


def tide_heights(ds, year, args):
    if args.tide_cache:
        file = args.tide_cache / f"{year}_tides.nc"
        if file.is_file():
            with xr.open_dataset(file) as cached:
                tides = cached["tide_m"]
                if (
                    np.array_equal(tides.x, ds.x)
                    and np.array_equal(tides.y, ds.y)
                    and np.isin(ds.time.values, tides.time.values).all()
                ):
                    print(f"Reusing {file}")
                    return tides.sel(time=ds.time).transpose("time", "y", "x").load()

    tides = pixel_tides(data=ds, model=TIDE_MODEL, directory=args.tide_model_dir,
                        resample=True).transpose("time", "y", "x")
    # Require matching coordinates before applying the tide mask.
    return xr.align(ds.ndvi, tides, join="exact")[1]


def make_composites(ndvi, tides, min_tide):
    """Compare the same valid NDVI observations, changing only the tide mask."""
    layers = {}
    for mode, stack in (("all_clear", ndvi), ("tide_filtered", ndvi.where(tides >= min_tide))):
        layers[f"{mode}_ndvi"] = stack.median("time", skipna=True)
        layers[f"{mode}_count"] = stack.count("time").astype("int16")
        layers[f"{mode}_stdev"] = stack.std("time", skipna=True)
        layers[f"{mode}_tide"] = tides.where(stack.notnull()).median("time", skipna=True)
    return xr.Dataset(layers)


def annual_composites(year, bbox, args):
    first, last = args.months
    start = pd.Timestamp(year, first, 1)
    end = pd.Timestamp(year, last, 1) + pd.offsets.MonthEnd()
    resolution = 10 if args.sensor == "sentinel2" else RESOLUTION
    cache_settings = (bbox, CRS, resolution, str(start), str(end), TIDE_MODEL, args.min_tide,
                      str(args.tide_model_dir.resolve()), str(args.tide_cache), 2)
    if args.sensor == "sentinel2":
        cache_settings += (args.sensor,)
    cache_key = hashlib.sha256(repr(cache_settings).encode()).hexdigest()[:16]
    cache = args.output / "composites" / f"{year}_{cache_key}.nc"
    if cache.is_file():
        with xr.open_dataset(cache) as saved:
            return saved.load().rio.write_crs(CRS)

    dem = load_dem_stac(bbox, output_crs=CRS, resolution=resolution).compute()
    loader = load_sentinel2_indices_stac if args.sensor == "sentinel2" else load_water_index_stac
    ds = loader(
        bbox,
        f"{start:%Y-%m-%d}/{end:%Y-%m-%d}",
        output_crs=CRS,
        resolution=resolution,
        dem=dem,
        mask_terrain_shadow=True,
        include_ndvi=True,
    )
    tides = tide_heights(ds, year, args).compute()
    ndvi = ds.ndvi.transpose("time", "y", "x").compute()
    result = make_composites(ndvi, tides, args.min_tide)
    diagnostics = summarise_observations(ndvi, tides, args.min_tide)
    # Keep only plain grid coordinates and arrays in the cache.
    saved = xr.Dataset(
        {name: (layer.dims, layer.values)
         for name, layer in result.data_vars.items()},
        coords={"x": result.x.values, "y": result.y.values},
    ).merge(diagnostics, join="exact")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache.parent) as temp:
        staged = Path(temp) / cache.name
        saved.to_netcdf(staged)
        staged.replace(cache)
    return saved.rio.write_crs(CRS)


def line_parts(geometry):
    if geometry.geom_type == "LineString" and not geometry.is_empty:
        yield geometry
    elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geometry.geoms:
            yield from line_parts(part)


def threshold_contours(ndvi, count, args):
    ndvi = ndvi.where(count >= args.min_count).rio.write_crs(CRS)
    values = ndvi.values
    valid = values[np.isfinite(values)]
    if valid.size and valid.min() < args.threshold < valid.max():
        # Disconnected or masked pixels may produce no contours.
        return subpixel_contours(ndvi, z_values=args.threshold, crs=CRS,
                                 min_vertices=2, errors="ignore", verbose=False)
    return gpd.GeoDataFrame(geometry=[], crs=CRS)


def extract_candidates(ndvi, count, corridor, args, year, mode, raw=None):
    if raw is None:
        raw = threshold_contours(ndvi, count, args)
    segments = []
    for line in raw.geometry:
        segments.extend(part for part in line_parts(line.intersection(corridor))
                        if part.length >= args.min_length)
    return gpd.GeoDataFrame(
        {"year": [year] * len(segments), "indicator": ["DEA_VEdge_candidate"] * len(segments),
         "sensor": [args.sensor] * len(segments),
         "tide_mode": [mode] * len(segments), "ndvi_threshold": [args.threshold] * len(segments),
         "min_observations": [args.min_count] * len(segments),
         "min_tide_m": [args.min_tide if mode == "tide_filtered" else None] * len(segments)},
        geometry=segments, crs=CRS,
    )


def compare_transects(transects, references, edges):
    coast = {mode: unary_union(frame.geometry) for mode, frame in edges.items()}
    records = []
    for date, group in references.groupby("Date"):
        reference = unary_union(group.geometry)
        for transect in transects.itertuples():
            crossing = transect.geometry.intersection(reference)
            if crossing.is_empty or crossing.geom_type != "Point":
                continue
            row = {"reference_date": date, "transect_id": transect.TransectID}
            for mode, candidates in coast.items():
                point = transect.geometry.intersection(candidates)
                row[f"{mode}_status"] = (
                    "missing" if point.is_empty else "unique" if point.geom_type == "Point"
                    else "ambiguous"
                )
                row[f"{mode}_offset_m"] = (
                    transect.geometry.project(point) - transect.geometry.project(crossing)
                    if row[f"{mode}_status"] == "unique" else np.nan
                )
            records.append(row)
    return pd.DataFrame(records, columns=[
        "reference_date", "transect_id", "all_clear_status", "all_clear_offset_m",
        "tide_filtered_status", "tide_filtered_offset_m",
    ])


def save_map(path, year, composites, edges, references, corridor, args):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), layout="constrained")
    bounds = corridor.bounds
    max_count = max(args.min_count, int(composites.all_clear_count.max()))
    for column, (mode, label) in enumerate(TIDE_MODES.items()):
        ndvi = composites[f"{mode}_ndvi"]
        count = composites[f"{mode}_count"]
        ndvi.where(count >= args.min_count).plot.imshow(
            ax=axes[0, column], vmin=-0.2, vmax=0.8, cmap="BrBG",
            cbar_kwargs={"label": "Median NDVI"},
        )
        count.astype(float).plot.imshow(
            ax=axes[1, column], vmin=0, vmax=max_count, cmap="viridis",
            cbar_kwargs={"label": "Valid observations"},
        )
        if not edges[mode].empty:
            edges[mode].plot(ax=axes[0, column], color="magenta", linewidth=1)
        if not references.empty:
            references.plot(ax=axes[0, column], color="black", linewidth=0.8, linestyle="--")
        for row in range(2):
            ax = axes[row, column]
            ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]), aspect="equal",
                   title=f"{label} · {'NDVI and contours' if row == 0 else 'coverage'}",
                   xlabel="BNG easting (m)", ylabel="BNG northing (m)")
    sensor_label = "Sentinel-2" if args.sensor == "sentinel2" else "Landsat"
    fig.suptitle(f"{sensor_label} {year}   NDVI {args.threshold:g}   Minimum tide {args.min_tide:+g} m rel. MSL"
                 "\nMagenta: contour candidates   Dashed black: mapped vegetation edges")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    aois = read_bng(args.aoi)
    area = unary_union(aois.geometry)
    bbox = tuple(float(value) for value in aois.to_crs("EPSG:4326").total_bounds)
    shorelines = read_bng(args.shorelines)
    coastal_buffers = shorelines.geometry.simplify(RESOLUTION / 6).buffer(args.corridor)
    corridor = unary_union(coastal_buffers).intersection(area)
    if corridor.is_empty:
        raise ValueError("The MSL shoreline search corridor does not overlap the AOI")
    references = read_bng(args.references).clip(corridor)
    dates = pd.to_datetime(references.Date, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise ValueError("Reference features within the corridor must have ISO dates (YYYY-MM-DD)")
    transects = read_bng(args.transects)
    transects = transects[transects.intersects(corridor)]

    args.output.mkdir(parents=True, exist_ok=True)
    for year in dict.fromkeys(args.years):
        print(f"Processing {year}")
        composites = annual_composites(year, bbox, args)
        year_dir = args.output / str(year)
        year_dir.mkdir(exist_ok=True)
        reference = references[(dates.dt.year == year)
                               & dates.dt.month.between(*args.months)]
        raw, edges = {}, {}
        with TemporaryDirectory(dir=year_dir) as temp:
            staged = Path(temp)
            for mode in TIDE_MODES:
                ndvi = composites[f"{mode}_ndvi"]
                count = composites[f"{mode}_count"]
                ndvi.rio.write_nodata(np.nan, encoded=True).rio.to_raster(staged / f"{mode}_ndvi.tif")
                count.rio.to_raster(staged / f"{mode}_count.tif")
                raw[mode] = threshold_contours(ndvi, count, args)
                edges[mode] = extract_candidates(ndvi, count, corridor, args, year, mode,
                                                 raw=raw[mode])
                (staged / f"{mode}_vegetation_edge.geojson").write_text(
                    edges[mode].to_crs("EPSG:4326").to_json(drop_id=True), encoding="utf-8"
                )
                print(f"  {mode}: {len(edges[mode])} candidate segments")
            comparison = compare_transects(transects, reference, edges)
            comparison.to_csv(staged / "transect_comparison.csv", index=False)
            paired = comparison.dropna(subset=["all_clear_offset_m", "tide_filtered_offset_m"])
            if len(paired):
                errors = paired[["all_clear_offset_m", "tide_filtered_offset_m"]].abs().mean()
                print(f"  Mean absolute offset on {len(paired)} paired transect-date matches: "
                      f"{errors.all_clear_offset_m:.1f} m all-clear, "
                      f"{errors.tide_filtered_offset_m:.1f} m tide-filtered")
            save_map(staged / "comparison.png", year, composites, edges, reference, corridor, args)
            save_diagnostics(staged, year, composites, raw, edges, reference, corridor,
                             comparison, args, TIDE_MODES)
            for product in staged.iterdir():
                product.replace(year_dir / product.name)


if __name__ == "__main__":
    main()
