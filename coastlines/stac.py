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


def load_water_index_stac(
    bbox,
    datetime,
    output_crs="EPSG:32630",
    resolution=30,
    cloud_cover=80,
    platforms=None,
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

    catalog = pystac_client.Client.open(
        PLANETARY_COMPUTER_STAC,
        modifier=planetary_computer.sign_inplace,
    )

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
    )

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

    return out