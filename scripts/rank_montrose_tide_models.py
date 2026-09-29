from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from eo_tides.validation import tide_correlation


# =====================================================================
# SETTINGS
# =====================================================================

# Representative offshore point at Montrose
lon = -2.45
lat = 56.73

# Use a long period to minimise tidal aliasing
time_range = (
    "2015",
    "2025",
)

# Only compare the two models we have tested
models = [
    "EOT20",
    "FES2022",
]

# Contains:
#   EOT20/
#   fes2022b/ocean_tide_20241025/
tide_model_dir = (
    "/home/mh322u/tide_models_scotland"
)

output_dir = Path(
    "outputs/montrose_tide_model_ranking"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# RUN TIDE-CORRELATION RANKING
#
# eo-tides will:
#
# 1. load Landsat imagery from Microsoft Planetary Computer
# 2. calculate satellite water-index observations
# 3. model tides at matching acquisition times
# 4. calculate pixel-wise tide/wetness correlations
# 5. rank the tide models using the same valid intertidal pixels
# =====================================================================

print()
print("=" * 70)
print("MONTROSE TIDE MODEL RANKING")
print("=" * 70)

print()
print(f"Location: {lon}, {lat}")
print(f"Period:   {time_range[0]}-{time_range[1]}")
print(f"Models:   {models}")
print()


corr_df, corr_da = tide_correlation(
    x=lon,
    y=lat,
    time=time_range,
    model=models,
    directory=tide_model_dir,

    # ~6 km diameter region centred on Montrose beach
    buffer=3000,

    # Landsat only, to match the Coastlines workflow
    load_ls=True,
    load_s2=False,

    # Moderate scene-level cloud threshold.
    # Pixel QA is still applied by the loader.
    cloud_cover=20,

    # Standard wet/dry NDWI threshold
    index_threshold=0.0,

    # Defaults used by eo-tides for identifying useful
    # dynamic/intertidal pixels
    freq_min=0.01,
    freq_max=0.99,
    corr_min=0.15,

    # Avoid unnecessary multiprocessing overhead for two models.
    parallel=False,
)


# =====================================================================
# RESULTS
# =====================================================================

print()
print("=" * 70)
print("MODEL RANKING")
print("=" * 70)

print()
print(corr_df)


# Sort best first
corr_sorted = corr_df.sort_values(
    "rank"
)


print()
print("Best model:")
print(
    corr_sorted.index[0]
)

print()

for model_name, row in corr_sorted.iterrows():

    print(
        f"{model_name:10s} "
        f"correlation = {row['correlation']:.4f}, "
        f"rank = {int(row['rank'])}"
    )


# =====================================================================
# SAVE TABLE
# =====================================================================

csv_file = (
    output_dir
    / "montrose_EOT20_FES2022_ranking.csv"
)

corr_df.to_csv(
    csv_file
)

print()
print(
    f"Saved ranking table: {csv_file}"
)


# =====================================================================
# FIGURE 1:
# OVERALL MODEL CORRELATION
# =====================================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)


plot_df = (
    corr_df
    .sort_values("correlation")
)


ax.bar(
    plot_df.index,
    plot_df["correlation"],
)


ax.set_ylabel(
    "Mean tide–inundation correlation"
)

ax.set_xlabel(
    "Tide model"
)

ax.set_title(
    "Montrose tide-model ranking"
)

ax.axhline(
    0,
    linewidth=0.8,
    linestyle="--",
)


# Add values above bars
for i, value in enumerate(
    plot_df["correlation"]
):

    ax.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )


fig.tight_layout()


bar_file = (
    output_dir
    / "montrose_EOT20_FES2022_ranking.png"
)


fig.savefig(
    bar_file,
    dpi=250,
    bbox_inches="tight",
)

plt.close(
    fig
)


# =====================================================================
# FIGURE 2:
# PIXEL-WISE CORRELATION MAPS
# =====================================================================

facet = corr_da.plot.imshow(
    col="tide_model",
    cmap="RdBu",
    vmin=-1.0,
    vmax=1.0,
    figsize=(11, 5),
)


facet.fig.suptitle(
    "Montrose satellite inundation–tide correlation",
    y=1.03,
)


map_file = (
    output_dir
    / "montrose_EOT20_FES2022_correlation_maps.png"
)


facet.fig.savefig(
    map_file,
    dpi=250,
    bbox_inches="tight",
)


plt.close(
    facet.fig
)


print()
print("=" * 70)
print("COMPLETE")
print("=" * 70)

print()
print(f"Ranking plot:     {bar_file}")
print(f"Correlation maps: {map_file}")
print()