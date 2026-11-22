cd ~/eac/t3/t3-methodes; date +%H:%M:%S
for n in PG_T85_f25_l814c PG_T85_f35_l814c PG_T85_f50_l814c PG_T85_f25_l1218c PG_T85_f30_l1218c PG_T85_f40_l1218c PG_T85_f30_l1624c PG_T85_f40_l1624c PGS30_err PGS30_oof PGS30_rand; do
  s=$(grep -c "^saved" logs/$n.log 2>/dev/null); st=$(tail -1 logs/$n.log 2>/dev/null | cut -c1-5); c=$( [ -s candidates/$n.pt ] && echo fichier || echo - )
  r=$(grep -E "^\| $n " diag/row_$n.md 2>/dev/null | cut -d'|' -f3,4 | cut -c1-60)
  echo "$n pas=$st saved=$s $c $r"
done
tail -2 ~/eac/t3/orchestrateur/sub_PG_T85_f20_l1218c.log 2>/dev/null | cut -c1-200
nvidia-smi --query-gpu=memory.used --format=csv,noheader
