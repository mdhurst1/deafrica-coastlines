#!/usr/bin/env python3
"""Extract summer vegetation edges from Landsat and Sentinel-2 at Montrose Bay."""

from calendar import monthrange
from datetime import date
from functools import cache
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from dea_tools.spatial import subpixel_contours
from odc.stac import load
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from coastlines.stac import (
    PLANETARY_COMPUTER_STAC,
    load_dem_stac,
    load_sentinel2_indices_stac,
    load_water_index_stac,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "pilot_inputs" / "montrose_vedge"
OUTPUT = ROOT.parent / "montrose_vedge_results"

YEARS = range(2018, 2026)
FIRST_MONTH, LAST_MONTH = 5, 9
NDVI_THRESHOLD = 0.20
MIN_OBSERVATIONS = 3
CORRIDOR_METRES = 300
MIN_LENGTH_METRES = 90

CRS = "EPSG:27700"
SENSORS = {
    "landsat": (30, load_water_index_stac),
    "sentinel2": (10, load_sentinel2_indices_stac),
}
WORLDCOVER_PALETTE = {
    10: (0, 100, 0), 20: (255, 187, 34), 30: (255, 255, 76),
    40: (240, 150, 255), 50: (250, 0, 0), 60: (180, 180, 180),
    70: (240, 240, 240), 80: (0, 100, 200), 90: (0, 150, 160),
    95: (0, 207, 117), 100: (250, 230, 160),
}


def worldcover_mask(bbox, output):
    """Save the full 10 m class raster and return its built-up and water polygons."""
    path = output / "worldcover_2021.tif"
    area_key = sha256(repr(bbox).encode()).hexdigest()[:16]
    if path.is_file():
        with rasterio.open(path) as raster:
            if raster.tags().get("aoi_key") == area_key:
                return exclusion_polygons(raster.read(1), raster.transform)

    catalog = pystac_client.Client.open(PLANETARY_COMPUTER_STAC)
    items = list(catalog.search(
        collections=["esa-worldcover"], bbox=bbox, datetime="2021-01-01/2021-12-31"
    ).items())
    if not items:
        raise RuntimeError("No ESA WorldCover 2021 tiles cover the Montrose AOI")

    landcover = load(
        items, bands=["map"], bbox=bbox, crs=CRS, resolution=10, resampling="nearest",
        groupby=lambda *_: 0, patch_url=planetary_computer.sign, fail_on_error=True,
    )["map"].isel(time=0)
    values = landcover.values
    valid = np.isfinite(values) & (values > 0)
    if not valid.any():
        raise RuntimeError("ESA WorldCover returned no valid pixels for Montrose")
    classes = np.where(valid, values, 0).astype("uint8")
    transform = landcover.rio.transform()
    with TemporaryDirectory(dir=output) as temp:
        staged = Path(temp) / path.name
        with rasterio.open(
            staged, "w", driver="GTiff", width=classes.shape[1], height=classes.shape[0],
            count=1, dtype="uint8", crs=CRS, transform=transform, nodata=0, compress="deflate",
        ) as raster:
            raster.write(classes, 1)
            raster.write_colormap(1, WORLDCOVER_PALETTE)
            raster.set_band_description(1, "ESA WorldCover 2021 land-cover class")
            raster.update_tags(aoi_key=area_key, class_50="Built-up", class_80="Permanent water bodies")
        staged.replace(path)
    return exclusion_polygons(classes, transform)


def exclusion_polygons(classes, transform):
    excluded = np.isin(classes, (50, 80))
    polygons = []
    for geometry, _ in shapes(excluded.astype("uint8"), mask=excluded, transform=transform):
        polygon = shape(geometry)
        polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
    return unary_union(polygons)


@cache
def dem_for_grid(bbox, resolution):
    return load_dem_stac(bbox, output_crs=CRS, resolution=resolution).compute()


def summer_composite(sensor, year, bbox, output):
    resolution, loader = SENSORS[sensor]
    start = date(year, FIRST_MONTH, 1)
    end = date(year, LAST_MONTH, monthrange(year, LAST_MONTH)[1])
    settings = (sensor, bbox, CRS, resolution, start.isoformat(), end.isoformat(), 1)
    key = sha256(repr(settings).encode()).hexdigest()[:16]
    path = output / "composites" / f"{sensor}_{year}_{key}.nc"
    if path.is_file():
        print(f"  Reusing {path.name}", flush=True)
        with xr.open_dataset(path) as saved:
            return saved.load()

    imagery = loader(
        bbox, f"{start}/{end}", output_crs=CRS, resolution=resolution,
        dem=dem_for_grid(bbox, resolution), mask_terrain_shadow=True, include_ndvi=True,
    )
    ndvi = imagery.ndvi.transpose("time", "y", "x").compute()
    composite = xr.Dataset({
        "ndvi": (("y", "x"), ndvi.median("time", skipna=True).values),
        "count": (("y", "x"), ndvi.count("time").values.astype("int32")),
    }, coords={"x": ndvi.x.values, "y": ndvi.y.values})

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        composite.to_netcdf(staged)
        staged.replace(path)
    return composite


def line_parts(geometry):
    if geometry.geom_type == "LineString" and not geometry.is_empty:
        yield geometry
    elif geometry.geom_type in ("MultiLineString", "GeometryCollection"):
        for part in geometry.geoms:
            yield from line_parts(part)


def vegetation_edges(composite, corridor):
    ndvi = composite.ndvi.where(composite["count"] >= MIN_OBSERVATIONS).rio.write_crs(CRS)
    values = ndvi.values
    valid = values[np.isfinite(values)]
    if not valid.size or not valid.min() < NDVI_THRESHOLD < valid.max():
        return []

    contours = subpixel_contours(
        ndvi, z_values=NDVI_THRESHOLD, crs=CRS, min_vertices=2, errors="ignore", verbose=False,
    )
    return [
        part
        for contour in contours.geometry
        for part in line_parts(contour.intersection(corridor))
        if part.length >= MIN_LENGTH_METRES
    ]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    aoi = gpd.read_file(INPUTS / "aoi.geojson")
    bbox = tuple(float(value) for value in aoi.to_crs("EPSG:4326").total_bounds)
    area = unary_union(aoi.to_crs(CRS).geometry)
    shorelines = gpd.read_file(INPUTS / "msl.geojson").to_crs(CRS)
    coastal_buffers = shorelines.geometry.simplify(5).buffer(CORRIDOR_METRES)
    corridor = unary_union(coastal_buffers).intersection(area)
    if corridor.is_empty:
        raise ValueError("The coastal search corridor does not overlap the Montrose AOI")

    exclusion = worldcover_mask(bbox, OUTPUT)
    with TemporaryDirectory(dir=OUTPUT) as temp:
        staging = Path(temp)
        for sensor in SENSORS:
            unmasked, masked = [], []
            for year in YEARS:
                print(f"{sensor} {year}", flush=True)
                composite = summer_composite(sensor, year, bbox, OUTPUT)
                lines = vegetation_edges(composite, corridor)
                properties = {"year": year, "sensor": sensor, "indicator": "DEA_VEdge_candidate",
                              "ndvi_threshold": NDVI_THRESHOLD, "min_observations": MIN_OBSERVATIONS}
                unmasked.extend({**properties, "geometry": line} for line in lines)
                masked.extend(
                    {**properties, "geometry": part}
                    for line in lines
                    for part in line_parts(line.difference(exclusion))
                )

            columns = ["year", "sensor", "indicator", "ndvi_threshold", "min_observations", "geometry"]
            for variant, records in (("unmasked", unmasked), ("masked", masked)):
                edges = gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=CRS)
                path = staging / f"{sensor}_vegetation_edges_{variant}.geojson"
                path.write_text(edges.to_crs("EPSG:4326").to_json(drop_id=True), encoding="utf-8")

        for path in sorted(staging.iterdir()):
            target = OUTPUT / path.name
            path.replace(target)
            print(f"Wrote {target}")


if __name__ == "__main__":
    main()
