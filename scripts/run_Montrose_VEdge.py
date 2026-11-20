#!/usr/bin/env python3
"""Extract summer vegetation edges from Landsat and Sentinel-2 at Montrose Bay."""

from calendar import monthrange
from datetime import date
from functools import cache
from hashlib import sha256
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from dea_tools.spatial import subpixel_contours
from pystac_client.exceptions import APIError
from rasterio.features import shapes
from rasterio.io import MemoryFile
from shapely.geometry import shape
from shapely.ops import unary_union

from coastlines.stac import (
    load_dem_stac,
    load_sentinel2_indices_stac,
    load_water_index_stac,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "pilot_inputs" / "montrose_vedge"
OUTPUT = ROOT.parent / "montrose_vedge_results"

YEARS = range(1988, 2026)
SENTINEL2_FIRST_YEAR = 2015
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
NATURESCOT_WCS = "https://ogc.nature.scot/geoserver/habitatsandspecies/wcs"
NATURESCOT_COVERAGE = "habitatsandspecies:HLCM_2024_EUNIS_LEVEL2"
# Raster values from the NatureScot Level 2 legend (P: Water; J: Built-up).
EXCLUDED_LAND_COVER = (1, 22)
NATURESCOT_PALETTE = {
    1: (32, 33, 218), 2: (132, 40, 0), 3: (140, 64, 0),
    4: (92, 63, 0), 5: (128, 148, 89), 6: (204, 202, 117),
    7: (227, 207, 118), 8: (170, 143, 165), 9: (172, 113, 125),
    10: (123, 91, 149), 11: (125, 70, 39), 12: (84, 39, 63),
    13: (57, 144, 100), 14: (55, 116, 41), 15: (0, 44, 20),
    16: (73, 105, 82), 17: (0, 73, 40), 18: (48, 156, 84),
    19: (144, 137, 76), 20: (179, 123, 60), 21: (215, 127, 63),
    22: (22, 23, 21), 24: (80, 80, 75), 25: (49, 195, 138),
    26: (255, 100, 255), 27: (255, 173, 232), 28: (255, 195, 186),
    29: (149, 111, 30),
}


def retry_catalog(label, operation, *args, **kwargs):
    """Retry transient HTTP responses from the STAC catalogue."""
    for delay in (5, 15, 30, None):
        try:
            return operation(*args, **kwargs)
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            if delay is None or status not in (429, 500, 502, 503, 504):
                raise
            print(f"  {label}: HTTP {status}. Retrying in {delay}s.", flush=True)
            sleep(delay)


def naturescot_mask(area_bounds, output):
    """Fetch and cache the NatureScot 2024 land-cover raster on its 10 m grid."""
    west, south, east, north = area_bounds
    bounds = (
        floor(west / 10) * 10, floor(south / 10) * 10,
        ceil(east / 10) * 10, ceil(north / 10) * 10,
    )
    path = output / "naturescot_landcover_2024.tif"
    if path.is_file():
        with rasterio.open(path) as raster:
            if raster.bounds == bounds and raster.tags().get("coverage") == NATURESCOT_COVERAGE:
                return exclusion_polygons(raster.read(1), raster.transform)

    query = urlencode({
        "service": "WCS", "version": "1.0.0", "request": "GetCoverage",
        "coverage": NATURESCOT_COVERAGE, "CRS": CRS, "bbox": ",".join(map(str, bounds)),
        "resx": 10, "resy": 10, "format": "GeoTIFF",
    })
    with urlopen(f"{NATURESCOT_WCS}?{query}", timeout=60) as response:
        data = response.read()

    if not data.startswith((b"MM\x00*", b"II*\x00")):
        raise RuntimeError("NatureScot WCS did not return a GeoTIFF")
    with MemoryFile(data) as memory, memory.open() as raster:
        if (raster.count != 1 or raster.crs != CRS
                or raster.res != (10, 10) or raster.bounds != bounds):
            raise RuntimeError("NatureScot WCS returned an unexpected raster grid")
        classes = raster.read(1, masked=True).filled(0)
        if not classes.any():
            raise RuntimeError("NatureScot WCS returned no classified pixels for Montrose")
        transform = raster.transform
        profile = raster.profile

    profile.update(nodata=0, compress="deflate")
    with TemporaryDirectory(dir=output) as temp:
        staged = Path(temp) / path.name
        with rasterio.open(staged, "w", **profile) as raster:
            raster.write(classes, 1)
            raster.write_colormap(1, NATURESCOT_PALETTE)
            raster.set_band_description(1, "NatureScot 2024 EUNIS Level 2 land-cover class")
            raster.update_tags(coverage=NATURESCOT_COVERAGE, class_1="Water", class_22="Built-up")
        staged.replace(path)
    print(f"Wrote {path}", flush=True)
    return exclusion_polygons(classes, transform)


def exclusion_polygons(classes, transform):
    excluded = np.isin(classes, EXCLUDED_LAND_COVER)
    polygons = []
    for geometry, _ in shapes(excluded.astype("uint8"), mask=excluded, transform=transform):
        polygon = shape(geometry)
        polygons.append(polygon if polygon.is_valid else polygon.buffer(0))
    return unary_union(polygons)


@cache
def dem_for_grid(bbox, resolution):
    dem = retry_catalog("DEM catalogue", load_dem_stac, bbox, output_crs=CRS, resolution=resolution)
    return dem.compute()


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

    dem = dem_for_grid(bbox, resolution)
    try:
        imagery = retry_catalog(
            f"{sensor} {year} catalogue", loader, bbox, f"{start}/{end}",
            output_crs=CRS, resolution=resolution, dem=dem, mask_terrain_shadow=True,
            include_ndvi=True,
        )
    except RuntimeError as exc:
        if not str(exc).startswith(("No Landsat scenes found for bbox=",
                                    "No Sentinel-2 L2A scenes found for bbox=")):
            raise
        print(f"  No scenes for {start}/{end}; skipping {sensor} {year}.", flush=True)
        return None
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

    exclusion = naturescot_mask(area.bounds, OUTPUT)
    with TemporaryDirectory(dir=OUTPUT) as temp:
        staging = Path(temp)
        for sensor in SENSORS:
            unmasked, masked = [], []
            for year in YEARS:
                if sensor == "sentinel2" and year < SENTINEL2_FIRST_YEAR:
                    continue
                print(f"{sensor} {year}", flush=True)
                composite = summer_composite(sensor, year, bbox, OUTPUT)
                if composite is None:
                    continue
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
            for variant, records in (("unmasked", unmasked), ("masked_naturescot", masked)):
                edges = gpd.GeoDataFrame(records, columns=columns, geometry="geometry", crs=CRS)
                path = staging / f"{sensor}_vegetation_edges_{variant}.geojson"
                path.write_text(edges.to_crs("EPSG:4326").to_json(drop_id=True), encoding="utf-8")

        for path in sorted(staging.iterdir()):
            target = OUTPUT / path.name
            path.replace(target)
            print(f"Wrote {target}")


if __name__ == "__main__":
    main()
