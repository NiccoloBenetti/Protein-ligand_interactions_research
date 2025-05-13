#!/usr/bin/env bash
set -euo pipefail

# Trova la directory in cui si trova questo script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Passa in testing_samples relativa allo script
TARGET_DIR="${SCRIPT_DIR}/testing_samples"

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Errore: directory testing_samples non trovata in ${SCRIPT_DIR}" >&2
  exit 1
fi

cd "${TARGET_DIR}"

# Loop su tutte le sottodirectory e rimuovi interactions.csv se esiste
for dir in */; do
  CSV_FILE="${dir}interactions.csv"
  if [[ -f "${CSV_FILE}" ]]; then
    echo "Rimuovo ${CSV_FILE}"
    rm "${CSV_FILE}"
  else
    echo "Nessun interactions.csv in ${dir}"
  fi
done
