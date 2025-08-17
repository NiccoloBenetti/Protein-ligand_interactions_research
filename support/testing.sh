#!/bin/bash

# --- 1) Calcola la directory in cui risiede questo script (support/) ---
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 2) Imposta root_dir a una cartella sopra script_dir (HPC-drugDiscovery) ---
root_dir="$(dirname "$script_dir")"

# --- 2bis) Rigenera l'eseguibile per essere sicuro di testare la versione aggiornata ---
"$script_dir/deleteCSV.sh"
rm -rf "$root_dir/build"
cmake -S "$root_dir" -B "$root_dir/build"
cmake --build "$root_dir/build"

# --- 3) Numero massimo di cartelle da processare (opzionale) ---
if [ -z "$1" ]; then
    max_dirs=-1
else
    max_dirs=$1
fi

count=0

# --- 4) Entra in testing_samples dentro support/ ---
cd "$script_dir/testing_samples" || { echo "Cannot cd into testing_samples"; exit 1; }

# --- 5) Loop su ogni sottocartella ---
for dir in */; do
    count=$((count + 1))
    if [ "$max_dirs" -gt 0 ] && [ "$count" -gt "$max_dirs" ]; then
        break
    fi

    cd "$dir" || { echo "Cannot cd into $dir"; cd ..; continue; }

    protein_file=$(find . -name '*_pocket.pdb' -print -quit)
    ligand_file=$(find . -name '*_ligand.mol2' -print -quit)

    if [[ -n "$protein_file" && -n "$ligand_file" ]]; then
        # Esegui interaction dal folder build in HPC-drugDiscovery
        "$root_dir/build/interaction" "$protein_file" "$ligand_file"
    else
        echo "Files missing in directory $dir"
    fi

    cd ..
done
