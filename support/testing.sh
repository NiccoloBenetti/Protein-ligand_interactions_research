#!/bin/bash

# Controlla se è stato fornito un numero massimo di directory da processare
if [ -z "$1" ]; then
    max_dirs=-1  # Se non è fornito un numero, setta max_dirs a -1 per indicare nessun limite
else
    max_dirs=$1
fi

count=0

# Salva la posizione base da cui parte lo script
BASE_DIR=$(pwd)
INTERACTION_EXECUTABLE="$BASE_DIR/build/interaction"

if [ ! -x "$INTERACTION_EXECUTABLE" ]; then
    echo "Errore: eseguibile $INTERACTION_EXECUTABLE non trovato o non eseguibile."
    exit 1
fi


# Loop attraverso tutte le sottodirectory
for dir in support/testing_samples/*/; do
    # Incrementa il contatore
    count=$((count + 1))
    
    # Controlla se il contatore ha raggiunto il massimo, solo se max_dirs è maggiore di 0
    if [ "$max_dirs" -gt 0 ] && [ "$count" -gt "$max_dirs" ]; then
        break
    fi
    
    # Entra nella directory
    pushd "$dir" > /dev/null || { echo "Impossibile entrare nella directory $dir"; exit 1; }

    # Trova il file della proteina e del ligando
    protein_file=$(find . -name '*_pocket.pdb' -print -quit)
    ligand_file=$(find . -name '*_ligand.mol2' -print -quit)
    
    # Controlla se entrambi i file esistono
    if [[ -n "$protein_file" && -n "$ligand_file" ]]; then
        # Esegui il tuo programma passando i due file come argomenti
        "$INTERACTION_EXECUTABLE" "$protein_file" "$ligand_file"
    else
        echo "Files missing in directory $dir"
    fi

    # Torna indietro
    popd > /dev/null

done

