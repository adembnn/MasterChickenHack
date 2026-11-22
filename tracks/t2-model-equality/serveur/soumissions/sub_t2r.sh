# session relais d Adem : envoi T2 avec reessai sur 429 (attente indiquee par le serveur), 5 essais au plus.
set -a; . ~/eac/.env; set +a
F=$1
for i in 1 2 3 4 5; do
  date -u +%H:%M:%S
  R=$(curl -s -m 600 -w "\nHTTP %{http_code}" -X POST "http://34.147.229.55/submit/33-model-equality" -H "X-API-Key: $EAC_API_KEY" -F "file=@$F;type=text/csv" | sed -E "s/[A-Za-z0-9_\-]{32,}/[MASQUE]/g")
  echo "$F"; echo "$R"
  echo "$R" | grep -q "HTTP 429" || break
  W=$(echo "$R" | grep -oE "[Ww]ait [0-9]+" | grep -oE "[0-9]+"); sleep $(( ${W:-60} + 2 ))
done
