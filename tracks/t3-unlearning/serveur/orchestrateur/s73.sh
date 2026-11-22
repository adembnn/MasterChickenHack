cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_RDt24_70_l1016c.pt > sub_PG_RDt24_70_l1016c.log 2>&1 &)
echo lance
cd ~/eac/t3; ls t3-methodes/cache | grep -i "RDt"; grep -rhn "RDt24" t3-redistill/*.sh t3-redistill/code/*.sh t3-redistill/tm/*.sh 2>/dev/null | cut -c1-400 | head -8
head -c 700 t3-methodes/logs/PG_RDt24_70_l915c.log 2>/dev/null | tr '\r' '\n' | head -2 | cut -c1-600; ls t3-redistill/tm 2>/dev/null | head; ls t3-redistill/tm/cache 2>/dev/null | grep RDt | head
