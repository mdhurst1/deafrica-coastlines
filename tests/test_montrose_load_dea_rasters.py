from coastlines.vector import load_rasters


# =====================================================================
# SETTINGS
# =====================================================================

raster_path = "outputs/dea_rasters"

raster_version = "scotland_v0.1"

study_area = "montrose"

start_year = 1988
end_year = 2025


# =====================================================================
# LOAD DEA RASTER STACK
# =====================================================================

yearly_ds, gapfill_ds = load_rasters(
    path=raster_path,
    raster_version=raster_version,
    study_area=study_area,
    water_index="mndwi",
    start_year=start_year,
    end_year=end_year,
)


# =====================================================================
# REPORT
# =====================================================================

print()
print("=" * 70)
print("ANNUAL DATASET")
print("=" * 70)

print(yearly_ds)

print()
print(
    "Years:",
    yearly_ds.year.values,
)

print()
print(
    "Variables:",
    list(yearly_ds.data_vars),
)


print()
print("=" * 70)
print("GAPFILL DATASET")
print("=" * 70)

print(gapfill_ds)

print()
print(
    "Years:",
    gapfill_ds.year.values,
)

print()
print(
    "Variables:",
    list(gapfill_ds.data_vars),
)


# =====================================================================
# BASIC CHECKS
# =====================================================================

assert yearly_ds.sizes["year"] == (
    end_year - start_year + 1
)

assert gapfill_ds.sizes["year"] == (
    end_year - start_year + 1
)

for var in [
    "mndwi",
    "count",
    "stdev",
]:

    assert var in yearly_ds

    assert var in gapfill_ds


print()
print(
    "All raster loading checks passed."
)