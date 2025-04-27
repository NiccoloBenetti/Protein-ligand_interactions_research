#!/bin/bash

# Cambia directory nella directory Testing, che si trova nella directory corrente
cd support/testing_samples

# Loop attraverso tutte le sottodirectory
for dir in */; do
    # Costruisce il percorso completo del file interaction.csv in ogni sottodirectory
    interaction_csv="${dir}interactions.csv"
    
    # Controlla se il file interaction.csv esiste
    if [[ -f "$interaction_csv" ]]; then
        # Elimina il file interaction.csv
        rm "$interaction_csv"
        echo "Removed $interaction_csv"
    else
        echo "No interactions.csv file in $dir"
    fi
done

