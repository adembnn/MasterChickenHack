# session relais : attend la fin du sub.sh en cours (pid en argument 1) puis envoie le fichier (argument 2)
cd ~/eac/t3/orchestrateur
while kill -0 "$1" 2>/dev/null; do sleep 2; done
N=$(basename "$2" .pt)
bash sub.sh "$2" > "sub_$N.log" 2>&1
