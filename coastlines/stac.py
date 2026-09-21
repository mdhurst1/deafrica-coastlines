import pystac_client
import planetary_computer
from odc.stac import load


PLANETARY_COMPUTER_STAC = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)


def load_water_index_stac(
    bbox,
    datetime,
    output_crs="EPSG:32630",
    resolution=30,
    cloud_cover=80,
):
    """
    Load Landsat Collection 2 Level-2 imagery from Microsoft
    Planetary Computer and calculate NDWI and MNDWI.

    Parameters
    ----------
    bbox : list
        Bounding box as [xmin, ymin, xmax, ymax] in WGS84.
    datetime : str
        STAC datetime range, e.g.
        "2020-01-01/2020-12-31".
    output_crs : str
        Output CRS.
    resolution : int
        Output pixel resolution in metres.
    cloud_cover : float
        Maximum scene-level cloud cover percentage.

    Returns
    -------
    xarray.Dataset
        Dataset containing `mndwi` and `ndwi`.
    """

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
            "platform": {"eq": "landsat-8"},
        },
    )

    items = search.item_collection()

    if len(items) == 0:
        raise RuntimeError(
            f"No Landsat scenes found for bbox={bbox}, "
            f"datetime={datetime}"
        )

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

    # Landsat Collection 2 Level-2 surface-reflectance scaling
    scale = 0.0000275
    offset = -0.2

    green = ds.green.where(ds.green != 0) * scale + offset
    nir = ds.nir08.where(ds.nir08 != 0) * scale + offset
    swir = ds.swir16.where(ds.swir16 != 0) * scale + offset

    # QA_PIXEL mask:
    # bit 0 = fill
    # bit 1 = dilated cloud
    # bit 2 = cirrus
    # bit 3 = cloud
    # bit 4 = cloud shadow
    # bit 5 = snow
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