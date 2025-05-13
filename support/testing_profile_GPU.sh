#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Uso:
#   ./testing.sh [N]        # esegue il profiling in modalità GPU
# --------------------------------------------------------------------------
shopt -s nullglob

# 0) parametro opzionale: massimo numero di directory da processare
max_dirs=${1:--1}
count=0

# 1) base + eseguibile
BASE_DIR=$(pwd)
INTERACTION_EXECUTABLE="$BASE_DIR/build/interaction"
[[ -x $INTERACTION_EXECUTABLE ]] || { echo "Errore: $INTERACTION_EXECUTABLE non trovato"; exit 1; }

# 2) configurazione fissa per GPU
TRACE="cuda,nvtx,osrt"
MODE="gpu"

# 3) cartella di output per i CSV NVTX
OUT_DIR="$BASE_DIR/support/performance/GPU"
mkdir -p "$OUT_DIR"

# 4) loop sulle sub-directory di testing_samples
for dir in support/testing_samples/*/; do
    echo "📂 Processing directory: $dir"  # <--- aggiungi questa riga
    # fermati se hai già processato max_dirs dir
    if [[ $max_dirs -gt 0 && $count -ge $max_dirs ]]; then
        echo "▶︎ Raggiunto il limite di $max_dirs cartelle"; break
    fi

    dir_name=$(basename "$dir")
    pushd "$dir" >/dev/null || { echo "Impossibile entrare in $dir"; exit 1; }

    protein_file=$(find . -name '*_pocket.pdb'  -print -quit)
    ligand_file=$( find . -name '*_ligand.mol2' -print -quit)

    if [[ -n $protein_file && -n $ligand_file ]]; then
        # 4a) profiling con Nsight Systems
        nsys profile \
             --trace="$TRACE" --sample=cpu \
             --output "${MODE}_run" \
             "$INTERACTION_EXECUTABLE" "$protein_file" "$ligand_file"

        # 4b) estrai NVTX Range Summary → CSV dentro support/nvtx_sums
        nsys stats \
            --report nvtxsum --format csv \
            -o "${OUT_DIR}/${dir_name}_${MODE}" \
            "${MODE}_run.nsys-rep"

        echo "✓ Salvato ${dir_name}_${MODE}_nvtxsum.csv"
    else
        echo "⚠️  File mancanti in $dir_name — salto"
    fi

    popd >/dev/null
    (( count++ ))
done

echo "▶︎ Completato: processate $count cartelle"