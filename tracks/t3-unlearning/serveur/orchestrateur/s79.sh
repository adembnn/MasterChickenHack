cd ~/eac/t3/orchestrateur
$HOME/eac/t3/.venv/bin/python mk_g10.py ~/eac/t3/t3-methodes/candidates/PG_RDm3t24_80_l1016c.pt ~/eac/t3/t3-methodes/candidates/G10_RDm3t24_80_l1016.pt | tail -1
(nohup bash -c "sleep 60; bash file.sh G10_RDm3t24_80_l1016" > file3.log 2>&1 &)
echo "sonde en file"; pgrep -u adem -fa "sub.sh" | cut -c1-100
