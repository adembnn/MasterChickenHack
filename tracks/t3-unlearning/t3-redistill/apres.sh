#!/bin/bash
# usage: apres.sh MOTIF N commande...   attend qu'il reste moins de N processus dont la ligne de commande contient MOTIF (expression reguliere), puis lance la commande
cd ~/eac/t3/t3-redistill; m=$1; n=$2; shift 2
while [ $(pgrep -u adem -fc -- "$m") -ge $n ]; do sleep 20; done
exec "$@"
