# session relais d Adem : envoi T2 (CSV). UNE tentative, pas de reessai automatique : le creneau de 30 min est partage avec Titouan.
set -a; . ~/eac/.env; set +a
F=$1
date -u +%H:%M:%S
R=$(curl -s -m 600 -w "\nHTTP %{http_code}" -X POST "http://34.147.229.55/submit/33-model-equality" -H "X-API-Key: $EAC_API_KEY" -F "file=@$F;type=text/csv" | sed -E "s/[A-Za-z0-9_\-]{32,}/[MASQUE]/g")
echo "$F"; echo "$R"
