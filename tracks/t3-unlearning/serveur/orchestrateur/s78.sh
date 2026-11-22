cd ~/eac/t3/orchestrateur
for p in $(pgrep -u adem -f "bash file.sh"); do kill $p; done
sleep 1; pgrep -u adem -fa "sub.sh" | cut -c1-110
(nohup bash -c "sleep 45; bash file.sh RDL_s3k_l6 PG_RDt24_70_l1218c" > file2.log 2>&1 &)
echo "file 2 lancee"
