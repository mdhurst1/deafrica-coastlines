# import modules
import pystac_client
import planetary_computer
from odc.stac import load


# Approximate Montrose Bay test area
bbox = [-2.52, 56.70, -2.42, 56.76]

# Open Microsoft Planetary Computer STAC catalogue
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search for Landsat 8 Collection 2 Level 2 scenes during 2020
search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=bbox,
    datetime="2020-01-01/2020-12-31",
    query={
        "eo:cloud_cover": {"lt": 50},
        "platform": {"eq": "landsat-8"},
    },
)

items = search.item_collection()

print(f"Found {len(items)} scenes")

# Show a few scenes
for item in items[:10]:
    print(
        item.id,
        item.datetime.date(),
        item.properties.get("eo:cloud_cover"),
    )

# Stop cleanly if nothing was found
if len(items) == 0:
    raise RuntimeError("No Landsat scenes found for the Montrose test area")

# Load the bands needed for water-index calculation
ds = load(
    items,
    bands=[
        "green",
        "nir08",
        "swir16",
        "qa_pixel",
    ],
    bbox=bbox,
    crs="EPSG:32630",
    resolution=30,
    groupby="solar_day",
    chunks={
        "time": 1,
        "x": 1024,
        "y": 1024,
    },
)

# ---------------------------------------------------------------------
# Convert Landsat Collection 2 DNs to surface reflectance
# ---------------------------------------------------------------------

scale = 0.0000275
offset = -0.2

# DN = 0 is the Landsat Collection 2 fill value
green = ds.green.where(ds.green != 0) * scale + offset
nir = ds.nir08.where(ds.nir08 != 0) * scale + offset
swir = ds.swir16.where(ds.swir16 != 0) * scale + offset


# ---------------------------------------------------------------------
# Mask cloud, cloud shadow, snow, cirrus and fill using QA_PIXEL
# Landsat 8/9 QA_PIXEL:
# bit 0 = fill
# bit 1 = dilated cloud
# bit 2 = cirrus
# bit 3 = cloud
# bit 4 = cloud shadow
# bit 5 = snow
# ---------------------------------------------------------------------

qa = ds.qa_pixel.fillna(0).astype("uint16")

bad_bits = (
    (1 << 0) |
    (1 << 1) |
    (1 << 2) |
    (1 << 3) |
    (1 << 4) |
    (1 << 5)
)

clear = (qa & bad_bits) == 0


# ---------------------------------------------------------------------
# Calculate water indices
# ---------------------------------------------------------------------

ndwi = (green - nir) / (green + nir)
mndwi = (green - swir) / (green + swir)

ndwi = ndwi.where(clear)
mndwi = mndwi.where(clear)

print("\nNDWI:")
print(ndwi)

print("\nMNDWI:")
print(mndwi)

import matplotlib.pyplot as plt

scene = mndwi.isel(time=0).compute()

print("\nPlotting scene:")
print(scene.time.values)

scene.plot(
    vmin=-1,
    vmax=1,
    cmap="RdBu"
)

plt.title(f"Montrose MNDWI - {scene.time.values}")
plt.savefig("montrose_mndwi.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved montrose_mndwi.png")