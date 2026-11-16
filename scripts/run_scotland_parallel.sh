#!/usr/bin/env bash

# Run the Scotland DEA Coastlines grid tile-by-tile in parallel.
#
# Intended location:
#   repository root
#
# Default usage:
#   ./run_scotland_parallel.sh
#
# Choose number of workers:
#   WORKERS=2 ./run_scotland_parallel.sh
#   WORKERS=4 ./run_scotland_parallel.sh
#   WORKERS=6 ./run_scotland_parallel.sh
#
# The driver is run WITHOUT --analyse so that expensive raster/shoreline
# extraction can proceed independently for each tile. Once all tiles are
# complete, run:
#
#   python -u scripts/run_scotland_grid_shorelines.py --all --analyse
#
# Existing completed tile outputs are reused by the Python driver.

set -uo pipefail

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

WORKERS="${WORKERS:-4}"
FIRST_TILE="${FIRST_TILE:-1}"
LAST_TILE="${LAST_TILE:-238}"

PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRIVER="${DRIVER:-${SCRIPT_DIR}/run_Scotland_grid.py}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/national_logs/${RUN_ID}}"

# Prevent each Python process from spawning many additional numerical threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Make variables available inside xargs worker shells.
export PYTHON_BIN
export DRIVER
export LOG_DIR

# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: WORKERS must be a positive integer."
    exit 1
fi

if ! [[ "${FIRST_TILE}" =~ ^[0-9]+$ && "${LAST_TILE}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: FIRST_TILE and LAST_TILE must be integers."
    exit 1
fi

if (( FIRST_TILE > LAST_TILE )); then
    echo "ERROR: FIRST_TILE must be <= LAST_TILE."
    exit 1
fi

if [[ ! -f "${DRIVER}" ]]; then
    echo "ERROR: Driver not found:"
    echo "  ${DRIVER}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

echo
echo "======================================================================"
echo "SCOTLAND DEA COASTLINES — PARALLEL GRID RUN"
echo "======================================================================"
echo "Driver:        ${DRIVER}"
echo "Python:        ${PYTHON_BIN}"
echo "Tiles:         ${FIRST_TILE}-${LAST_TILE}"
echo "Workers:       ${WORKERS}"
echo "Log directory: ${LOG_DIR}"
echo "Started:       $(date)"
echo "======================================================================"
echo

seq "${FIRST_TILE}" "${LAST_TILE}" |
    xargs -P "${WORKERS}" -n 1 bash -c '
        tile="$1"
        log="${LOG_DIR}/tile_${tile}.log"

        echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting tile ${tile}"

        "${PYTHON_BIN}" -u "${DRIVER}" --tiles "${tile}" > "${log}" 2>&1
        exit_code=$?

        if (( exit_code != 0 )); then
            echo "[$(date "+%Y-%m-%d %H:%M:%S")] FAILED tile ${tile} (exit ${exit_code}) — see ${log}"

        elif grep -Eq '^Failed:[[:space:]]+0[[:space:]]*$' "${log}" \
            && grep -Eq '^Successful/already complete:[[:space:]]+1[[:space:]]*$' "${log}"; then

            echo "[$(date "+%Y-%m-%d %H:%M:%S")] Finished tile ${tile}"

        else
            echo "[$(date "+%Y-%m-%d %H:%M:%S")] FAILED tile ${tile} — see ${log}"
        fi
    ' _

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

echo
echo "======================================================================"
echo "PARALLEL GRID RUN FINISHED"
echo "======================================================================"
echo "Finished:      $(date)"
echo "Log directory: ${LOG_DIR}"
echo

failed_tiles=()

while IFS= read -r logfile; do
    tile="$(basename "${logfile}" .log | sed "s/^tile_//")"

    if grep -Eq '^Failed:[[:space:]]+0[[:space:]]*$' "${logfile}" \
       && grep -Eq '^Successful/already complete:[[:space:]]+1[[:space:]]*$' "${logfile}"; then

        # Tile genuinely completed successfully.
        continue

    else
        failed_tiles+=("${tile}")
    fi

done < <(
    find "${LOG_DIR}" \
        -maxdepth 1 \
        -type f \
        -name 'tile_*.log' \
        | sort -V
)

total=$((LAST_TILE - FIRST_TILE + 1))
failed=${#failed_tiles[@]}
successful=$((total - failed))

echo "Tiles requested: ${total}"
echo "No error found:  ${successful}"
echo "Failed:          ${failed}"

if (( failed > 0 )); then
    echo
    echo "Failed tile numbers:"
    printf "  %s\n" "${failed_tiles[@]}"

    echo
    echo "Rerun failed tiles individually with, for example:"
    echo "  ${PYTHON_BIN} -u ${DRIVER} --tiles <tile_no>"
else
    echo
    echo "No failed tiles were detected in the logs."
fi

echo
echo "After all extraction tiles are complete, run the single analysis/merge pass:"
echo
echo "  ${PYTHON_BIN} -u ${DRIVER} --all --analyse"
echo
