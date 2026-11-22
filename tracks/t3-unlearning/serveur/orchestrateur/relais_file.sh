# session relais : UNE seule chaine d envoi. Attend la fin du pid (arg 1), puis envoie chaque fichier restant dans l ordre.
cd ~/eac/t3/orchestrateur
P=$1; shift
while kill -0 "$P" 2>/dev/null; do sleep 2; done
for F in "$@"; do N=$(basename "$F" .pt); bash sub.sh "$F" > "sub_$N.log" 2>&1; done
