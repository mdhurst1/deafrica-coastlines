from coastlines.stac import load_water_index_stac


bbox = [-2.52, 56.70, -2.42, 56.76]

ds = load_water_index_stac(
    bbox=bbox,
    datetime="2020-01-01/2020-12-31",
)

print(ds)

scene = ds.mndwi.isel(time=0).compute()

import matplotlib.pyplot as plt

scene.plot(
    vmin=-1,
    vmax=1,
    cmap="RdBu",
)

plt.savefig(
    "montrose_mndwi.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()