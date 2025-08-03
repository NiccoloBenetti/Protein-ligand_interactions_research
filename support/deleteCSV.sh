#!/bin/bash
cd support/testing_samples || { echo "Cartella testing_samples non trovata"; exit 1; }

for dir in */ ; do
    interaction_csv="${dir}interactions.csv"
    rep=( ${dir}*.nsys-rep )
    nvtx_csvs=( ${dir}*_nvtxsum.csv )
    sqlite=( ${dir}*.sqlite )

    for file in "$interaction_csv" "$rep" "$sqlite" "${nvtx_csvs[@]}"; do             
        [[ -f $file ]] && { rm "$file"; echo "Removed $file"; }
    done
done
    