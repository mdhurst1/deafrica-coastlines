import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from eo_tides.model import model_tides


# Approximate offshore point at Montrose Bay
x = -2.42
y = 56.72

# One month of hourly tides
start_date = "2020-01-01"
end_date = "2020-02-01"

times = pd.date_range(
    start_date,
    end_date,
    freq="1h",
    inclusive="left",
)

tides = model_tides(
    x=x,
    y=y,
    time=times,
    model="EOT20",
    directory="/home/mh322u/tide_models",
)

print(tides.head())

plot_time = tides.index.get_level_values("time")
plot_height = tides["tide_height"]


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    plot_time,
    plot_height,
    linewidth=1,
)

ax.axhline(
    0,
    linewidth=0.8,
)

ax.set_xlabel("Date")
ax.set_ylabel("Tide height relative to mean sea level (m)")

ax.set_title(
    "Modelled tide at Montrose Bay — EOT20\n"
    "1 January–31 January 2020"
)

# Major labels every 5 days
ax.xaxis.set_major_locator(
    mdates.DayLocator(interval=5)
)

# Explicit day-month-year labels
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b %Y")
)

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig(
    "montrose_tide_curve_january_2020.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved montrose_tide_curve_january_2020.png"
)