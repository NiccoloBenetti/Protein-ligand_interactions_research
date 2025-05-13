#!/bin/bash
cd support/testing_samples || { echo "Cartella testing_samples non trovata"; exit 1; }

for dir in */ ; do
    interaction_csv="${dir}interactions.csv"

    # file di profilazione (gpu / cpu)
    gpu_rep=(${dir}*_run.nsys-rep)
    nvtx_csvs=(${dir}*_nvtxsum.csv)
    gpu_sqlite=(${dir}*_run.sqlite)

    for file in "$interaction_csv" "$gpu_rep" "$cpu_rep" "$gpu_csv" "$cpu_csv" \
                "$gpu_sqlite" "$cpu_sqlite" "$gpu_nvtxcsv" "${nvtx_csvs[@]}"; do             
        [[ -f $file ]] && { rm "$file"; echo "Removed $file"; }
    done
done
    