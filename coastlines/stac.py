# import modules
import numpy as np
import xarray as xr

from dea_tools.spatial import hillshade
from skimage.morphology import binary_dilation, binary_opening, disk

import pystac_client
import planetary_computer
from odc.stac import load
from collections import Counter

PLANETARY_COMPUTER_STAC = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

DEFAULT_PLATFORMS = [
    "landsat-5",
    "landsat-7",
    "landsat-8",
    "landsat-9",
]

def add_solar_angles(ds, items):
    """
    Attach Landsat solar elevation and azimuth to an xarray
    Dataset loaded using odc.stac.

    Solar angles are extracted from the STAC item metadata and
    matched to the corresponding loaded timestep.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset returned by odc.stac.load().
    items : pystac.ItemCollection
        Landsat STAC items used to construct the dataset.

    Returns
    -------
    xarray.Dataset
        Dataset with `sun_elevation` and `sun_azimuth`
        variables along the time dimension.
    """

    item_metadata = []

    for item in items:

        sun_elevation = item.properties.get(
            "view:sun_elevation"
        )

        sun_azimuth = item.properties.get(
            "view:sun_azimuth"
        )

        if sun_elevation is None or sun_azimuth is None:
            print(
                f"Warning: solar angles missing for "
                f"{item.id}"
            )
            continue

        item_metadata.append(
            {
                "time": np.datetime64(item.datetime),
                "sun_elevation": float(sun_elevation),
                "sun_azimuth": float(sun_azimuth),
            }
        )

    if len(item_metadata) == 0:
        raise RuntimeError(
            "No solar-angle metadata found in "
            "the Landsat STAC items."
        )

    sun_elevations = []
    sun_azimuths = []

    for time in ds.time.values:

        # Find the STAC item closest in time to the
        # timestep produced by odc.stac.load().
        differences = np.array(
            [
                abs(meta["time"] - time)
                for meta in item_metadata
            ],
            dtype="timedelta64[s]",
        )

        closest_index = np.argmin(differences)

        sun_elevations.append(
            item_metadata[closest_index][
                "sun_elevation"
            ]
        )

        sun_azimuths.append(
            item_metadata[closest_index][
                "sun_azimuth"
            ]
        )

    ds["sun_elevation"] = xr.DataArray(
        sun_elevations,
        dims=["time"],
        coords={"time": ds.time},
    )

    ds["sun_azimuth"] = xr.DataArray(
        sun_azimuths,
        dims=["time"],
        coords={"time": ds.time},
    )

    return ds

def load_water_index_stac(
    bbox,
    datetime,
    output_crs="EPSG:32630",
    resolution=30,
    cloud_cover=80,
    platforms=None,
    fail_on_error=True,
    dem=None,
    mask_terrain_shadow=False,
    terrain_shadow_threshold=0.5,
    terrain_shadow_radius=1,
):
    """
    Load Landsat Collection 2 Level-2 imagery and calculate
    NDWI and MNDWI.

    Supports Landsat 5, 7, 8 and 9.

    Parameters
    ----------
    bbox : list
        [xmin, ymin, xmax, ymax] in WGS84.
    datetime : str
        Date range, e.g. "1987-01-01/2025-12-31".
    output_crs : str
        Output CRS.
    resolution : int
        Output resolution in metres.
    cloud_cover : float
        Maximum scene-level cloud cover percentage.
    platforms : list, optional
        Landsat platforms to use.

    Returns
    -------
    xarray.Dataset
        Dataset containing `mndwi` and `ndwi`.
    """

    if platforms is None:
        platforms = DEFAULT_PLATFORMS

    catalog = pystac_client.Client.open(PLANETARY_COMPUTER_STAC)

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=datetime,
        query={
            "eo:cloud_cover": {"lt": cloud_cover},
            "platform": {"in": platforms},
            "landsat:collection_category": {"eq": "T1"},
        },
    )

    items = search.item_collection()

    if len(items) == 0:
        raise RuntimeError(
            f"No Landsat scenes found for bbox={bbox}, "
            f"datetime={datetime}"
        )

    # Useful diagnostic
    platform_counts = Counter(
        item.properties.get("platform", "unknown")
        for item in items
    )

    print(f"Found {len(items)} Landsat scenes:")
    for platform, count in sorted(platform_counts.items()):
        print(f"  {platform}: {count}")

    ds = load(
        items,
        bands=[
            "green",
            "nir08",
            "swir16",
            "qa_pixel",
        ],
        bbox=bbox,
        crs=output_crs,
        resolution=resolution,
        groupby="solar_day",
        chunks={
            "time": 1,
            "x": 1024,
            "y": 1024,
        },
        patch_url=planetary_computer.sign,
        fail_on_error=fail_on_error,
    )

    # retrieve the solar angles from the STAC items and attach to the dataset
    ds = add_solar_angles(ds, items)

    # Collection 2 Level-2 surface-reflectance scaling
    scale = 0.0000275
    offset = -0.2

    green = ds.green.where(ds.green != 0) * scale + offset
    nir = ds.nir08.where(ds.nir08 != 0) * scale + offset
    swir = ds.swir16.where(ds.swir16 != 0) * scale + offset

    # Mask invalid surface-reflectance values.
    # This mirrors the original DEA Coastlines workflow.
    green = green.where(
        (green >= 0) & (green <= 1)
    )

    nir = nir.where(
        (nir >= 0) & (nir <= 1)
    )

    swir = swir.where(
        (swir >= 0) & (swir <= 1)
    )

    # QA_PIXEL
    #
    # bit 0: fill
    # bit 1: dilated cloud
    # bit 2: cirrus (L8/9; unused for earlier sensors)
    # bit 3: cloud
    # bit 4: cloud shadow
    # bit 5: snow
    qa = ds.qa_pixel.fillna(0).astype("uint16")

    bad_bits = (
        (1 << 0)
        | (1 << 1)
        | (1 << 2)
        | (1 << 3)
        | (1 << 4)
        | (1 << 5)
    )

    clear = (qa & bad_bits) == 0

    mndwi = ((green - swir) / (green + swir)).where(clear)
    ndwi = ((green - nir) / (green + nir)).where(clear)

    out = ds[[]].copy()

    out["mndwi"] = mndwi
    out["ndwi"] = ndwi

    out["sun_elevation"] = ds.sun_elevation
    out["sun_azimuth"] = ds.sun_azimuth

    if mask_terrain_shadow:

        if dem is None:
            raise ValueError(
                "A DEM must be supplied when "
                "mask_terrain_shadow=True"
            )

        out = terrain_shadow_masking_stac(
            ds=out,
            dem=dem,
            threshold=terrain_shadow_threshold,
            radius=terrain_shadow_radius,
        )

    return out

def load_dem_stac(
    bbox,
    output_crs="EPSG:32630",
    resolution=30,
):
    catalog = pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC
    )

    search = catalog.search(
        collections=["cop-dem-glo-30"],
        bbox=bbox,
    )

    items = search.item_collection()

    if len(items) == 0:
        raise RuntimeError(
            f"No Copernicus DEM found for bbox={bbox}"
        )

    dem = load(
        items,
        bands=["data"],
        bbox=bbox,
        crs=output_crs,
        resolution=resolution,
        groupby="id",
        patch_url=planetary_computer.sign,
    )

    # Mosaic DEM tiles
    elevation = dem.data.max(
        dim="time",
        skipna=True,
    )

    return elevation.rename("elevation")

def terrain_shadow_stac(
    dem,
    sun_elevation,
    sun_azimuth,
    threshold=0.5,
    radius=1,
):
    """
    Calculate a terrain-shadow mask from a DEM and solar geometry.

    This is the STAC equivalent of the original DEA Coastlines
    `terrain_shadow` function in raster.py.

    Terrain illumination is calculated using hillshade. Pixels with
    illumination below `threshold` are classified as terrain shadow.
    Morphological opening removes small isolated shadow pixels, and
    dilation expands the remaining shadow regions slightly to mask
    shadow edges.

    Parameters
    ----------
    dem : xarray.DataArray or numpy.ndarray
        2D Digital Elevation Model.

    sun_elevation : float
        Solar elevation angle in degrees above the horizon.

    sun_azimuth : float
        Solar azimuth angle in degrees clockwise from north.

    threshold : float, optional
        Hillshade illumination value below which pixels are considered
        terrain shadow. Default is 0.5, matching the original DEA
        Coastlines workflow.

    radius : int, optional
        Radius of the morphological disk used for opening and dilation.
        Default is 1.

    Returns
    -------
    xarray.DataArray
        Boolean terrain-shadow mask with dimensions ("y", "x").
        True indicates terrain shadow.
    """

    # Convert DEM to a numpy array if supplied as an xarray DataArray
    if isinstance(dem, xr.DataArray):
        dem_values = dem.values
        y = dem.y
        x = dem.x
    else:
        dem_values = np.asarray(dem)
        y = None
        x = None

    # Calculate illumination using the same hillshade function
    # as the original DEA Coastlines workflow
    hs = hillshade(
        dem_values,
        sun_elevation,
        sun_azimuth,
    )

    # Identify poorly illuminated terrain
    shadow = hs < threshold

    # Remove small isolated areas
    shadow = binary_opening(
        shadow,
        disk(radius),
    )

    # Expand remaining shadow slightly to include shadow edges
    shadow = binary_dilation(
        shadow,
        disk(radius),
    )

    # Preserve DEM coordinates where available
    if y is not None and x is not None:
        shadow = xr.DataArray(
            shadow,
            dims=("y", "x"),
            coords={
                "y": y,
                "x": x,
            },
            name="terrain_shadow",
        )

    else:
        shadow = xr.DataArray(
            shadow,
            dims=("y", "x"),
            name="terrain_shadow",
        )

    return shadow

def terrain_shadow_masking_stac(
    ds,
    dem,
    threshold=0.5,
    radius=1,
):
    """
    Calculate and apply terrain-shadow masks to a STAC-loaded
    Landsat water-index dataset.

    A separate terrain-shadow mask is calculated for every Landsat
    observation using that observation's solar elevation and azimuth.

    Terrain-shadow pixels are set to NaN in NDWI and MNDWI.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing:

            mndwi(time, y, x)
            ndwi(time, y, x)
            sun_elevation(time)
            sun_azimuth(time)

    dem : xarray.DataArray
        DEM on the same x/y grid as the Landsat dataset.

    threshold : float, optional
        Hillshade illumination threshold below which pixels are
        classified as terrain shadow. Default is 0.5.

    radius : int, optional
        Morphological opening/dilation radius. Default is 1.

    Returns
    -------
    xarray.Dataset
        Dataset with terrain-shadow pixels masked from NDWI and MNDWI,
        plus a boolean `terrain_shadow` variable with dimensions
        (time, y, x).
    """

    # Basic checks before doing any processing
    required_variables = [
        "sun_elevation",
        "sun_azimuth",
    ]

    for variable in required_variables:
        if variable not in ds:
            raise ValueError(
                f"Dataset does not contain required "
                f"variable '{variable}'"
            )

    # Ensure DEM spatial dimensions match Landsat
    if dem.sizes["y"] != ds.sizes["y"]:
        raise ValueError(
            "DEM and Landsat dataset have different y dimensions: "
            f"{dem.sizes['y']} vs {ds.sizes['y']}"
        )

    if dem.sizes["x"] != ds.sizes["x"]:
        raise ValueError(
            "DEM and Landsat dataset have different x dimensions: "
            f"{dem.sizes['x']} vs {ds.sizes['x']}"
        )

    shadow_masks = []

    # Calculate a terrain-shadow mask independently for
    # every satellite observation
    for i in range(ds.sizes["time"]):

        sun_elevation = float(
            ds["sun_elevation"]
            .isel(time=i)
            .values
        )

        sun_azimuth = float(
            ds["sun_azimuth"]
            .isel(time=i)
            .values
        )

        shadow = terrain_shadow_stac(
            dem=dem,
            sun_elevation=sun_elevation,
            sun_azimuth=sun_azimuth,
            threshold=threshold,
            radius=radius,
        )

        # Attach the acquisition time to this 2D mask
        shadow = shadow.expand_dims(
            time=[
                ds.time.values[i]
            ]
        )

        shadow_masks.append(shadow)

    # Combine all 2D masks into:
    #
    # terrain_shadow(time, y, x)
    terrain_shadow = xr.concat(
        shadow_masks,
        dim="time",
    )

    terrain_shadow = terrain_shadow.assign_coords(
        time=ds.time
    )

    terrain_shadow.name = "terrain_shadow"

    # Add mask to dataset so we can inspect/debug it later
    ds["terrain_shadow"] = terrain_shadow

    # Apply terrain-shadow masking
    if "mndwi" in ds:
        ds["mndwi"] = ds["mndwi"].where(
            ~terrain_shadow
        )

    if "ndwi" in ds:
        ds["ndwi"] = ds["ndwi"].where(
            ~terrain_shadow
        )

    return ds