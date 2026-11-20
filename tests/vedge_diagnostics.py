"""Diagnostics for the seasonal Montrose vegetation-edge pilot."""

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from rasterio.features import geometry_mask


def summarise_observations(ndvi, tides, min_tide):
    """Keep per-date coverage and a small sample of scenes for plotting."""
    records = []
    for scene_ndvi, scene_tide in zip(ndvi.values, tides.values):
        valid = np.isfinite(scene_ndvi)
        modelled_tides = scene_tide[np.isfinite(scene_tide)]
        records.append((
            valid.mean(),
            (valid & (scene_tide >= min_tide)).mean(),
            np.median(modelled_tides) if modelled_tides.size else np.nan,
        ))
    clear, accepted, median_tide = np.asarray(records).T
    suitable = np.flatnonzero(np.isfinite(median_tide) & (clear >= 0.2))
    if not suitable.size:
        suitable = np.flatnonzero(np.isfinite(median_tide) & (clear > 0))
    ordered = suitable[np.argsort(median_tide[suitable], kind="stable")]
    ranks = np.linspace(0, len(ordered) - 1, min(len(ordered), 5), dtype=int)
    samples = ordered[ranks]
    if len(ordered) > len(samples):
        # Keep examples near the decision boundary as well as the tide extremes.
        closest = ordered[np.abs(median_tide[ordered] - min_tide).argmin()]
        if closest not in samples:
            samples[2] = closest
            samples = samples[np.argsort(median_tide[samples], kind="stable")]

    return xr.Dataset(
        {
            "clear_fraction": ("date", clear),
            "retained_fraction": ("date", accepted),
            "scene_median_tide": ("date", median_tide),
            "example_ndvi": (("scene", "y", "x"), ndvi.values[samples]),
            "example_tide": (("scene", "y", "x"), tides.values[samples]),
        },
        coords={"date": ndvi.time.values, "scene": ndvi.time.values[samples],
                "x": ndvi.x.values, "y": ndvi.y.values},
    )


def _save(fig, path, title):
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _map_panel(ax, raster, bounds, title, cmap, limits, label):
    raster.astype(float).plot.imshow(
        ax=ax, cmap=cmap, vmin=limits[0], vmax=limits[1],
        cbar_kwargs={"label": label, "shrink": 0.72},
    )
    ax.set(xlim=(bounds[0], bounds[2]), ylim=(bounds[1], bounds[3]),
           aspect="equal", title=title, xlabel="BNG easting (m)", ylabel="BNG northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=7)
    ax.set_facecolor("#e8e8e8")


def _observation_plots(folder, composites, bounds, title, cutoff):
    days = pd.to_datetime(composites.date.values)
    summary = pd.DataFrame({
        "acquired_utc": days,
        "clear_grid_fraction": composites.clear_fraction.values,
        "retained_grid_fraction": composites.retained_fraction.values,
        "median_modelled_tide_m_rel_msl": composites.scene_median_tide.values,
    })
    summary.to_csv(folder / "qa_observations.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    axes[0].scatter(days, summary.median_modelled_tide_m_rel_msl, s=15, color="#234f79")
    axes[0].axhline(cutoff, color="#b95434", linestyle="--", label=f"Filter cutoff ({cutoff:+g} m)")
    axes[0].set(ylabel="Median modelled tide (m rel. MSL)")
    axes[0].legend()
    axes[1].plot(days, summary.clear_grid_fraction * 100, ".-", label="Clear NDVI")
    axes[1].plot(days, summary.retained_grid_fraction * 100, ".-",
                 label="Clear NDVI at or above cutoff")
    axes[1].set(ylabel="Share of processing grid (%)", ylim=(0, 105), xlabel="Acquisition date (UTC)")
    axes[1].legend()
    _save(fig, folder / "qa_01_acquisition_tides.png", f"{title} · Acquisitions and tidal filtering")

    scenes = composites.scene.values
    if not len(scenes):
        fig, ax = plt.subplots(figsize=(8, 2), layout="constrained")
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No scenes with both clear NDVI and modelled tide heights",
                ha="center", va="center", transform=ax.transAxes)
    else:
        fig, axes = plt.subplots(len(scenes), 3, squeeze=False, figsize=(12, 3.25 * len(scenes)),
                                 layout="constrained")
        for day, row in zip(scenes, axes):
            ndvi = composites.example_ndvi.sel(scene=day, drop=True)
            tide = composites.example_tide.sel(scene=day, drop=True)
            stamp = pd.Timestamp(day).strftime("%d %b %Y %H:%M UTC")
            _map_panel(row[0], ndvi, bounds, f"{stamp}\nClear NDVI", "BrBG", (-0.2, 0.8), "NDVI")
            _map_panel(row[1], tide, bounds, "Modelled tide", "RdBu_r", (-3, 3), "m rel. MSL")
            _map_panel(row[2], ndvi.where(tide >= cutoff), bounds,
                       f"NDVI after cutoff at {cutoff:+g} m", "BrBG", (-0.2, 0.8), "NDVI")
    _save(fig, folder / "qa_02_example_scenes.png", f"{title} · Example scenes from low to high tide")


def _composite_plot(folder, ds, bounds, title, min_count):
    all_count, tide_count = ds.all_clear_count, ds.tide_filtered_count
    comparable = (all_count >= min_count) & (tide_count >= min_count)
    difference = (ds.tide_filtered_ndvi - ds.all_clear_ndvi).where(comparable)
    retention = 100 * tide_count / all_count.where(all_count > 0)
    fig, axes = plt.subplots(3, 2, figsize=(11, 12), layout="constrained")
    panels = [
        (difference, f"NDVI shift (≥{min_count} observations in each)", "BrBG", (-0.2, 0.2), "Filtered minus all clear"),
        (retention, "Clear observations retained", "viridis", (0, 100), "%"),
        (ds.all_clear_stdev, "NDVI spread · all clear", "magma", (0, 0.4), "NDVI SD"),
        (ds.tide_filtered_stdev, "NDVI spread · tide-filtered", "magma", (0, 0.4), "NDVI SD"),
        (ds.all_clear_tide, "Median contributing tide · all clear", "RdBu_r", (-3, 3), "m rel. MSL"),
        (ds.tide_filtered_tide, "Median contributing tide · tide-filtered", "RdBu_r", (-3, 3), "m rel. MSL"),
    ]
    for ax, (raster, label, cmap, limits, units) in zip(axes.flat, panels):
        _map_panel(ax, raster, bounds, label, cmap, limits, units)
    _save(fig, folder / "qa_03_composite_differences.png", f"{title} · Effect of the tidal filter")


def _contour_plot(folder, ds, raw, edges, references, corridor, title, args, modes):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), height_ratios=(3, 1), layout="constrained")
    inside = geometry_mask([corridor], out_shape=ds.all_clear_ndvi.shape,
                           transform=ds.rio.transform(), invert=True)
    for column, (mode, label) in enumerate(modes.items()):
        ndvi = ds[f"{mode}_ndvi"].where(ds[f"{mode}_count"] >= args.min_count)
        panel = axes[0, column]
        _map_panel(panel, ndvi, corridor.bounds, label, "BrBG", (-0.2, 0.8), "NDVI")
        panel.set_xlim(corridor.bounds[0] - 120, corridor.bounds[2] + 120)
        panel.set_ylim(corridor.bounds[1] - 120, corridor.bounds[3] + 120)
        for x, y in _boundary_coords(corridor.boundary):
            panel.plot(x, y, color="#ffffff", linewidth=0.8)
        if not raw[mode].empty:
            raw[mode].plot(ax=panel, color="#e79b36", linewidth=1.2)
        if not edges[mode].empty:
            edges[mode].plot(ax=panel, color="magenta", linewidth=1.4)
        if not references.empty:
            references.plot(ax=panel, color="black", linewidth=0.8, linestyle="--")
        samples = ndvi.values[inside & np.isfinite(ndvi.values)]
        hist = axes[1, column]
        if samples.size:
            hist.hist(samples, bins=np.linspace(-1, 1, 81), color="#345c68")
        hist.axvline(args.threshold, linestyle="--", color="#af175f")
        hist.set(xlabel=f"NDVI in corridor ({samples.size:,} pixels)", ylabel="Count", xlim=(-1, 1))
    _save(fig, folder / "qa_04_threshold_and_contours.png",
          f"{title} · NDVI threshold {args.threshold:g} and contour screening"
          "\nWhite: corridor    Orange: raw contours    Magenta: retained    Dashed black: references")


def _boundary_coords(boundary):
    parts = boundary.geoms if boundary.geom_type == "MultiLineString" else (boundary,)
    for part in parts:
        yield part.xy


def _transect_plot(folder, comparison, title, modes):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    if comparison.empty:
        for ax in axes:
            ax.set_axis_off()
        axes[0].text(0.5, 0.5, "No reference transect crossings in this season",
                     ha="center", va="center", transform=axes[0].transAxes)
    else:
        statuses = ["unique", "missing", "ambiguous"]
        for column, (mode, label) in enumerate(modes.items()):
            counts = comparison[f"{mode}_status"].value_counts().reindex(statuses, fill_value=0)
            axes[0].bar(np.arange(3) + (column - 0.5) * 0.35, counts, width=0.35, label=label)
        axes[0].set(xticks=np.arange(3), xticklabels=statuses, ylabel="Transect–date pairs",
                    title="Candidate crossings")
        axes[0].legend()
        paired = comparison[["all_clear_offset_m", "tide_filtered_offset_m"]].dropna()
        if paired.empty:
            axes[1].set_axis_off()
            axes[1].text(0.5, 0.5, "No paired unique candidate crossings",
                         ha="center", va="center", transform=axes[1].transAxes)
        else:
            limit = max(1, float(np.abs(paired.to_numpy()).max())) * 1.1
            axes[1].plot([-limit, limit], [-limit, limit], linestyle="--", color="0.5")
            axes[1].scatter(paired.all_clear_offset_m, paired.tide_filtered_offset_m, s=12, alpha=0.65)
            axes[1].set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
                        xlabel="All-clear offset (m)", ylabel="Tide-filtered offset (m)",
                        title=f"Signed offsets to reference ({len(paired):,} pairs)")
    _save(fig, folder / "qa_05_transect_agreement.png", f"{title} · Comparison with reference lines")


def save_diagnostics(folder, year, composites, raw, edges, references, corridor, comparison, args, modes):
    sensor = "Sentinel-2" if args.sensor == "sentinel2" else "Landsat"
    title = f"{sensor} {year}"
    _observation_plots(folder, composites, corridor.bounds, title, args.min_tide)
    _composite_plot(folder, composites, corridor.bounds, title, args.min_count)
    _contour_plot(folder, composites, raw, edges, references, corridor, title, args, modes)
    _transect_plot(folder, comparison, title, modes)
