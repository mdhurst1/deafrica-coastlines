from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt

from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter, MaxNLocator

from coastlines.vector import load_rasters


# =====================================================================
# SETTINGS
# =====================================================================

raster_path = "outputs/dea_rasters"
raster_version = "scotland_v0.1"
study_area = "montrose"

start_year = 1988
end_year = 2025

water_index = "mndwi"

shoreline_file = Path(
    "outputs/montrose_vectors/"
    "montrose_DEA_MSL_annual_shorelines_1988_2025.geojson"
)

output_dir = Path("outputs/montrose_vector_QA")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "montrose_DEA_MSL_shorelines_magma_zoom.png"


# =====================================================================
# ZOOM WINDOW
# =====================================================================

zoom_bounds = (
    534000,       # xmin
    534500,       # xmax
    6_286_650,    # ymin
    6_288_100,    # ymax
)


# =====================================================================
# LOAD RASTERS
# =====================================================================

print()
print("=" * 70)
print("LOADING DEA RASTERS")
print("=" * 70)

yearly_ds, _ = load_rasters(
    path=raster_path,
    raster_version=raster_version,
    study_area=study_area,
    water_index=water_index,
    start_year=start_year,
    end_year=end_year,
)

raster_crs = yearly_ds.rio.crs

if raster_crs is None:
    raise ValueError("Raster dataset has no CRS.")

print(f"Raster CRS: {raster_crs}")


# =====================================================================
# LOAD SHORELINES
# =====================================================================

print()
print("=" * 70)
print("LOADING SHORELINES")
print("=" * 70)

shorelines = gpd.read_file(shoreline_file)

print(f"Loaded {len(shorelines)} shoreline segments.")
print(f"Input CRS: {shorelines.crs}")

shorelines = shorelines.to_crs(raster_crs)
shorelines["year"] = shorelines["year"].astype(int)

if "certainty" not in shorelines.columns:
    print()
    print("WARNING: no certainty attribute found.")
    shorelines["certainty"] = "good"

print()
print("Certainty classes:")
print(shorelines["certainty"].value_counts(dropna=False))


# =====================================================================
# YEARS AND COLOUR RAMP
# =====================================================================

years = sorted(shorelines["year"].unique())

norm = Normalize(vmin=min(years), vmax=max(years))
cmap = colormaps["magma"]


# =====================================================================
# BACKGROUND MNDWI
# =====================================================================

background_year = 2020
background = yearly_ds[water_index].sel(year=background_year)


# =====================================================================
# CERTAINTY STYLES
# =====================================================================

certainty_styles = {
    "good": {
        "linestyle": "-",
        "linewidth": 1.20,
        "alpha": 0.95,
    },
    "unstable data": {
        "linestyle": "--",
        "linewidth": 1.00,
        "alpha": 0.55,
    },
    "insufficient data": {
        "linestyle": ":",
        "linewidth": 1.00,
        "alpha": 0.40,
    },
}

default_style = {
    "linestyle": "--",
    "linewidth": 1.0,
    "alpha": 0.45,
}


# =====================================================================
# HELPERS
# =====================================================================

def plot_all_shorelines(ax, linewidth_scale=1.0):
    for year in years:
        yearly = shorelines[shorelines["year"] == year]
        colour = cmap(norm(year))

        for certainty, subset in yearly.groupby("certainty", dropna=False):
            style = certainty_styles.get(certainty, default_style)

            subset.plot(
                ax=ax,
                color=colour,
                linewidth=style["linewidth"] * linewidth_scale,
                linestyle=style["linestyle"],
                alpha=style["alpha"],
                zorder=5,
            )


def tidy_axes(ax, xbins=5, ybins=5):
    """
    Make axis labels cleaner:
    - no scientific notation
    - no 1e6 offset text
    - fewer tick labels
    """

    fmtx = ScalarFormatter(useOffset=False)
    fmtx.set_scientific(False)

    fmty = ScalarFormatter(useOffset=False)
    fmty.set_scientific(False)

    ax.xaxis.set_major_formatter(fmtx)
    ax.yaxis.set_major_formatter(fmty)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins))

    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)


# =====================================================================
# CREATE FIGURE
# =====================================================================

print()
print("=" * 70)
print("CREATING FIGURE")
print("=" * 70)

fig = plt.figure(figsize=(13.5, 6.8))

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[1.45, 0.72, 0.05],
    wspace=0.18,
)

ax_full = fig.add_subplot(gs[0])
ax_zoom = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])


# =====================================================================
# FULL CONTEXT PANEL
# =====================================================================

background.plot(
    ax=ax_full,
    cmap="Greys",
    vmin=-1,
    vmax=1,
    alpha=0.28,
    add_colorbar=False,
    zorder=1,
)

plot_all_shorelines(ax_full, linewidth_scale=0.72)

zxmin, zxmax, zymin, zymax = zoom_bounds

zoom_rectangle = Rectangle(
    (zxmin, zymin),
    zxmax - zxmin,
    zymax - zymin,
    fill=False,
    edgecolor="black",
    linewidth=1.5,
    linestyle="--",
    zorder=10,
)

ax_full.add_patch(zoom_rectangle)

ax_full.set_title("Montrose DEA-MSL annual shorelines", fontsize=14)
ax_full.set_xlabel("Easting (m)")
ax_full.set_ylabel("Northing (m)")
ax_full.set_aspect("equal")

tidy_axes(ax_full, xbins=6, ybins=6)


# =====================================================================
# ZOOM PANEL
# =====================================================================

background.plot(
    ax=ax_zoom,
    cmap="Greys",
    vmin=-1,
    vmax=1,
    alpha=0.22,
    add_colorbar=False,
    zorder=1,
)

plot_all_shorelines(ax_zoom, linewidth_scale=1.05)

ax_zoom.set_xlim(zxmin, zxmax)
ax_zoom.set_ylim(zymin, zymax)

ax_zoom.set_title("Zoomed beach section", fontsize=14)
ax_zoom.set_xlabel("Easting (m)")
ax_zoom.set_ylabel("Northing (m)")
ax_zoom.set_aspect("equal")

# Fewer ticks here to avoid crowding
tidy_axes(ax_zoom, xbins=4, ybins=5)


# =====================================================================
# YEAR COLOURBAR
# =====================================================================

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Year")

cbar.set_ticks([1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025])


# =====================================================================
# CERTAINTY LEGEND
# =====================================================================

legend_handles = [
    Line2D(
        [0], [0],
        color="0.25",
        linewidth=1.6,
        linestyle="-",
        alpha=0.95,
        label="Good",
    ),
    Line2D(
        [0], [0],
        color="0.25",
        linewidth=1.6,
        linestyle="--",
        alpha=0.65,
        label="Unstable data",
    ),
    Line2D(
        [0], [0],
        color="0.25",
        linewidth=1.6,
        linestyle=":",
        alpha=0.50,
        label="Insufficient data",
    ),
]

ax_zoom.legend(
    handles=legend_handles,
    title="DEA certainty",
    loc="upper left",
    frameon=True,
)


# =====================================================================
# TIGHTEN LAYOUT / SAVE
# =====================================================================

fig.subplots_adjust(
    left=0.06,
    right=0.96,
    bottom=0.10,
    top=0.92,
    wspace=0.20,
)

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print()
print("Saved figure:")
print(output_file)
print()
print("Done.")