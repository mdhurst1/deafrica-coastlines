import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from eo_tides.model import model_tides
from coastlines.stac import load_water_index_stac


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

bbox = [-2.52, 56.70, -2.42, 56.76]

# Approximate offshore point in Montrose Bay
tide_x = -2.42
tide_y = 56.72

tide_model_dir = "/home/mh322u/tide_models"

platforms = [
    "landsat-7",
    "landsat-8",
]


# ---------------------------------------------------------------------
# Load Landsat observations for the whole of 2020
# ---------------------------------------------------------------------

print("Finding Landsat observations during 2020...")

ds = load_water_index_stac(
    bbox=bbox,
    datetime="2020-01-01/2020-12-31",
    platforms=platforms,
)

landsat_times_all = pd.DatetimeIndex(ds.time.values)

if len(landsat_times_all) == 0:
    raise RuntimeError(
        "No Landsat observations found for Montrose in 2020."
    )

print(f"\nFound {len(landsat_times_all)} observations in 2020:")

for time in landsat_times_all:
    print(f"  {time}")


# ---------------------------------------------------------------------
# Find the month with the most Landsat observations
# ---------------------------------------------------------------------

months = landsat_times_all.to_period("M")

month_counts = (
    pd.Series(months)
    .value_counts()
    .sort_index()
)

print("\nObservations per month:")
print(month_counts)

best_month = month_counts.idxmax()

print(
    f"\nMonth with most observations: "
    f"{best_month} ({month_counts.loc[best_month]} scenes)"
)

month_start = best_month.start_time
month_end = (best_month + 1).start_time

in_month = (
    (landsat_times_all >= month_start)
    & (landsat_times_all < month_end)
)

landsat_times_month = landsat_times_all[in_month]

print("\nUsing acquisitions in selected month:")

for time in landsat_times_month:
    print(f"  {time}")

if len(landsat_times_month) == 0:
    raise RuntimeError(
        f"No Landsat observations found in selected month {best_month}."
    )


# ---------------------------------------------------------------------
# Model tides for ALL 2020 Landsat acquisition times
#
# We use all 2020 acquisitions to define the approximate Coastlines
# tidal window, rather than only the handful in the selected month.
# ---------------------------------------------------------------------

print("\nModelling tide heights for all 2020 Landsat acquisitions...")

landsat_tides_all = model_tides(
    x=tide_x,
    y=tide_y,
    time=landsat_times_all,
    model="EOT20",
    directory=tide_model_dir,
)

landsat_plot_time_all = landsat_tides_all.index.get_level_values("time")
landsat_height_all = landsat_tides_all["tide_height"]

print("\nAll 2020 Landsat acquisition tides:")

for time, height in zip(landsat_plot_time_all, landsat_height_all):
    print(f"  {time}: {height:.2f} m")


# ---------------------------------------------------------------------
# Calculate simple DEA Coastlines-style tidal cutoffs
#
# DEA Coastlines keeps observations centred on 0 m using the middle
# 50% of the tide range sampled by the satellite observations.
# ---------------------------------------------------------------------

tide_min = landsat_height_all.min()
tide_max = landsat_height_all.max()

tide_range = tide_max - tide_min
cutoff_buffer = tide_range * 0.25

tide_cutoff_min = -cutoff_buffer
tide_cutoff_max = cutoff_buffer

print("\nSatellite-observed tidal range in 2020:")
print(f"  Minimum: {tide_min:.2f} m")
print(f"  Maximum: {tide_max:.2f} m")

print("\nApproximate DEA Coastlines tidal cutoffs:")
print(f"  Lower: {tide_cutoff_min:.2f} m")
print(f"  Upper: {tide_cutoff_max:.2f} m")


# ---------------------------------------------------------------------
# Generate continuous hourly tide curve for selected month
# ---------------------------------------------------------------------

print(f"\nModelling hourly tides for {best_month}...")

hourly_times = pd.date_range(
    start=month_start,
    end=month_end,
    freq="1h",
    inclusive="left",
)

hourly_tides = model_tides(
    x=tide_x,
    y=tide_y,
    time=hourly_times,
    model="EOT20",
    directory=tide_model_dir,
)

hourly_plot_time = hourly_tides.index.get_level_values("time")
hourly_height = hourly_tides["tide_height"]


# ---------------------------------------------------------------------
# Model tide heights for Landsat acquisitions in selected month only
# ---------------------------------------------------------------------

print("\nModelling tide heights at Landsat acquisition times in selected month...")

landsat_tides_month = model_tides(
    x=tide_x,
    y=tide_y,
    time=landsat_times_month,
    model="EOT20",
    directory=tide_model_dir,
)

landsat_plot_time_month = landsat_tides_month.index.get_level_values("time")
landsat_height_month = landsat_tides_month["tide_height"]

print("\nSelected-month Landsat acquisition tides:")

for time, height in zip(landsat_plot_time_month, landsat_height_month):
    print(f"  {time}: {height:.2f} m")


# ---------------------------------------------------------------------
# Determine which selected-month observations fall inside the cutoff
# ---------------------------------------------------------------------

within_cutoff = (
    (landsat_height_month >= tide_cutoff_min)
    & (landsat_height_month <= tide_cutoff_max)
)

print(
    f"\nObservations inside tidal window: "
    f"{within_cutoff.sum()} / {len(landsat_height_month)}"
)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 7))

# Continuous tide curve
ax.plot(
    hourly_plot_time,
    hourly_height,
    linewidth=1,
    label="EOT20 modelled tide",
)

# Mean sea level
ax.axhline(
    0,
    linewidth=0.8,
    linestyle="--",
    label="Mean sea level",
)

# Approximate DEA Coastlines tidal acceptance window
ax.axhspan(
    tide_cutoff_min,
    tide_cutoff_max,
    alpha=0.15,
    label="Approx. Coastlines tidal window",
)

# Landsat acquisitions retained
ax.scatter(
    landsat_plot_time_month[within_cutoff],
    landsat_height_month[within_cutoff],
    s=70,
    marker="o",
    zorder=5,
    label="Landsat retained",
)

# Landsat acquisitions rejected
ax.scatter(
    landsat_plot_time_month[~within_cutoff],
    landsat_height_month[~within_cutoff],
    s=70,
    marker="x",
    zorder=5,
    label="Landsat rejected",
)

# Vertical lines marking Landsat acquisitions
for time in landsat_plot_time_month:
    ax.axvline(
        time,
        linewidth=0.5,
        alpha=0.3,
    )


# ---------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------

ax.set_xlabel("Date")
ax.set_ylabel("Tide height relative to mean sea level (m)")

ax.set_title(
    "Landsat acquisitions and modelled tides — Montrose Bay\n"
    f"{best_month.strftime('%B %Y')} | EOT20"
)

ax.xaxis.set_major_locator(
    mdates.DayLocator(interval=5)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b %Y")
)

fig.autofmt_xdate()

ax.legend()

plt.tight_layout()

output_file = (
    f"montrose_landsat_tides_"
    f"{best_month.strftime('%Y_%m')}.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"\nSaved {output_file}")
print("\nDone.")