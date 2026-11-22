#!/bin/bash
# usage: retry.sh LOG commande...   relance tant que la commande echoue faute de memoire (carte partagee souvent pleine)
L=$1; shift
for i in $(seq 1 90); do "$@" > "$L" 2>&1 && exit 0; grep -q -i -E "out of memory|ALLOC_FAILED|CUDA error" "$L" || exit 1; sleep 20; done
