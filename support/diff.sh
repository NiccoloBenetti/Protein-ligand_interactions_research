#!/usr/bin/env bash
# diff.sh
#
# Naviga l'albero dei test come testing.sh,
# copia ogni interactions.csv rinominandolo con il nome della directory che lo contiene
# in diff/CPU o diff/GPU dentro la cartella support (dove è questo script),
# poi confronta i file .csv omonimi e salva il report in diff/diff.txt.
#
# USO:
#   ./diff.sh --mode cpu   # default
#   ./diff.sh --mode gpu
#   ./diff.sh --mode cpu --root /path/to/project

set -euo pipefail

###############
# PARSING CLI #
###############
MODE="cpu"
ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$( tr '[:upper:]' '[:lower:]' <<< "${2}" )"
      shift 2
      ;;
    --root)
      ROOT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--mode cpu|gpu] [--root PATH]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[[ "$MODE" =~ ^(cpu|gpu)$ ]] || { echo "MODE must be cpu or gpu" >&2; exit 1; }

# Directory dello script (support folder)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Radice del progetto (default: un livello sopra support)
if [[ -z "$ROOT" ]]; then
  ROOT="$( realpath "$SCRIPT_DIR/.." )"
fi

# Cartella diff dentro support
DIFF_DIR="$SCRIPT_DIR/diff"
CPU_DIR="$DIFF_DIR/CPU"
GPU_DIR="$DIFF_DIR/GPU"
DIFF_FILE="$DIFF_DIR/diff.txt"

# Crea cartelle diff, CPU e GPU
mkdir -p "$CPU_DIR" "$GPU_DIR"

echo ">>> Modalità: $MODE"
DEST_DIR="$DIFF_DIR/$( tr '[:lower:]' '[:upper:]' <<< "$MODE" )"
echo ">>> Raccolgo i risultati in: $DEST_DIR"

#########################################
# 1. COPIA interactions.csv RINOMINATO  #
#########################################
while IFS= read -r -d '' CSV; do
  DIRNAME="$( basename "$( dirname "$CSV" )" )"
  cp -f "$CSV" "$DEST_DIR/${DIRNAME}.csv"
done < <( find "$ROOT" -type f -name "interactions.csv" -not -path "$DIFF_DIR/*" -print0 )

echo ">>> Copia completata."

###################################################
# 2. CONFRONTA E SCRIVI diff.txt                   #
###################################################
: > "$DIFF_FILE"

# Procedi solo se ci sono CSV in entrambe le cartelle
if compgen -G "$CPU_DIR/*.csv" > /dev/null && compgen -G "$GPU_DIR/*.csv" > /dev/null; then
  echo ">>> Confronto CSV tra CPU e GPU..."
  for CPU_CSV in "$CPU_DIR"/*.csv; do
    FILE="$( basename "$CPU_CSV" )"
    GPU_CSV="$GPU_DIR/$FILE"
    if [[ -f "$GPU_CSV" ]]; then
      if ! DIFF_OUTPUT=$( diff -u "$CPU_CSV" "$GPU_CSV" ); then
        {
          echo "=============================================="
          echo "Differenze in: $FILE"
          echo "----------------------------------------------"
          echo "$DIFF_OUTPUT"
          echo
        } >> "$DIFF_FILE"
      fi
    else
      echo "[MISSING] $FILE presente in CPU ma non in GPU" >> "$DIFF_FILE"
    fi
  done
  # File presenti solo in GPU
  for GPU_CSV in "$GPU_DIR"/*.csv; do
    FILE="$( basename "$GPU_CSV" )"
    [[ -f "$CPU_DIR/$FILE" ]] || echo "[MISSING] $FILE presente in GPU ma non in CPU" >> "$DIFF_FILE"
  done
  echo ">>> Report diff salvato in: $DIFF_FILE"
else
  echo ">>> Nessun confronto: assicurati di avere CSV in entrambe $CPU_DIR e $GPU_DIR" >&2
fi

