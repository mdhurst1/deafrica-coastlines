#!/usr/bin/env bash
# Build DEA Scotland rates PMTiles from the original EPSG:27700 GeoPackage.
# GeoJSONSeq is streamed: no large intermediate GeoJSON files are created.
# Requirements: ogr2ogr (GDAL) and Tippecanoe >= 2.17 with PMTiles output.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./build_dea_pmtiles.sh /path/to/Scotland_DEA_MSL_multiscale_rates_baseline2024_BNG.gpkg /path/to/CoastalMappingViewer [1000|300|30|all]

Start with 1000 (recommended). The 30 m archive may exceed GitHub hosting limits
when all annual distance attributes are included; check its size before committing.
EOF
}

if (( $# < 2 || $# > 3 )); then usage; exit 2; fi
GPKG="$1"
CMV_REPO="$2"
CHOICE="${3:-1000}"
[[ -f "$GPKG" ]] || { echo "ERROR: GeoPackage not found: $GPKG" >&2; exit 2; }
[[ -f "$CMV_REPO/package.json" ]] || { echo "ERROR: Expected CMV repo folder with package.json: $CMV_REPO" >&2; exit 2; }
command -v ogr2ogr >/dev/null || { echo "ERROR: ogr2ogr is missing (install GDAL)" >&2; exit 2; }
command -v tippecanoe >/dev/null || { echo "ERROR: tippecanoe is missing (version >=2.17 required for direct .pmtiles output)" >&2; exit 2; }

case "$CHOICE" in
  1000|300|30) SCALES=("$CHOICE") ;;
  all) SCALES=(1000 300 30) ;;
  *) usage; exit 2 ;;
esac

OUTDIR="${CMV_REPO%/}/public/data/DEA_Scotland"
mkdir -p "$OUTDIR"

build_one() {
  local scale="$1" min_zoom max_zoom output layer fields selection bytes
  layer="rates_${scale}m"
  output="$OUTDIR/Scotland_rates_${scale}m.pmtiles"
  case "$scale" in
    1000) min_zoom=8;  max_zoom=11 ;;
    300)  min_zoom=11; max_zoom=13 ;;
    30)   min_zoom=13; max_zoom=16 ;;
  esac

  # Only retain attributes needed for CMV symbols and rate/time-series popups.
  # 30 m uses point_id instead of hotspot_id.
  fields=(indicator baseline_year aggregation n_source_points)
  if [[ "$scale" == 30 ]]; then
    fields+=(point_id tile_id certainty sce nsm)
  else
    fields+=(hotspot_id)
  fi
  local period year
  for period in 1988_2025 2000_2025 2015_2025; do
    fields+=("rate_${period}" "se_${period}" "sig_${period}" "valid_obs_${period}")
  done
  for ((year=1988; year<=2025; year++)); do
    fields+=("dist_${year}")
  done
  selection="$(IFS=,; echo "${fields[*]}")"

  echo "=== Building $layer (zoom $min_zoom–$max_zoom) ==="
  echo "Source: $GPKG"
  echo "Target: $output"

  # In Tippecanoe: -r 1 plus -B at the minimum zoom stops thinning by zoom.
  # -pf -pk prevent implicit loss of features due to tile limits. Large tiles
  # remain possible; do not rely on these archives for scientific calculations.
  ogr2ogr \
    -f GeoJSONSeq /vsistdout/ "$GPKG" "$layer" \
    -t_srs EPSG:4326 \
    -select "$selection" \
    -lco RS=NO \
    -lco COORDINATE_PRECISION=6 \
    | tippecanoe \
        -f \
        -o "$output" \
        -l "$layer" \
        -Z "$min_zoom" \
        -z "$max_zoom" \
        -B "$min_zoom" \
        -r 1 \
        -pf \
        -pk \
        -pg

  bytes="$(wc -c < "$output")"
  echo "Created: $(du -h "$output" | cut -f1) ($bytes bytes)"
  if (( bytes >= 90000000 )); then
    echo "WARNING: This archive is near/above GitHub's ~100 MB per-file Git limit."
    echo "Do not commit it yet. We should move full time-series fields to sidecar"
    echo "files or host the PMTiles archive outside the repository."
  fi
  echo
}

for scale in "${SCALES[@]}"; do
  build_one "$scale"
done

echo "Done. Outputs are in: $OUTDIR"
echo "These files are not wired into CMV yet. We should inspect the output"
echo "sizes and test the first archive before replacing the working 5 km layer."
