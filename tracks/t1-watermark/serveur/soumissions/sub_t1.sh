set -a; . ~/eac/.env; set +a
F=$1
for i in 1 2 3 4 5 6; do
  date +%H:%M:%S
  R=$(curl -s -m 600 -w "\nHTTP %{http_code}" -X POST "http://34.147.229.55/submit/34-watermark-localisation-image" -H "X-API-Key: $EAC_API_KEY" -F "file=@$F;type=application/octet-stream" | sed -E 's/[A-Za-z0-9_\-]{32,}/[MASQUÉ]/g')
  echo "$F"; echo "$R"
  echo "$R" | grep -q "HTTP 429" || break
  W=$(echo "$R" | grep -oE "Wait [0-9]+" | grep -oE "[0-9]+"); sleep $(( ${W:-60} + 3 ))
done
