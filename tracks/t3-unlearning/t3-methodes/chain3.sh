#!/bin/bash
# h200e, apres la vague en cours : (1) PGr = recette gagnante + vues fortes tournees ; (2) PD2 = ancre vues degradees mais oubli aussi sur les vues fortes
cd ~/eac/t3/t3-methodes; export OMP_NUM_THREADS=4
until [ $(cat logs/PG_T85_f20_l1218c.log logs/PG_T85_f20_l1624c.log 2>/dev/null | grep -c "^saved") -ge 2 ]; do sleep 10; done
go() { n=$1; t=$2; shift 2; T="./py code/train.py --steps 1500 --wce 1 --lam 3 $*"
  (nohup code/retry.sh logs/$n.log $T --name $n --tgt $t --fset all > /dev/null 2>&1 < /dev/null &); (nohup code/finish.sh $n $t > /dev/null 2>&1 < /dev/null &)
  (nohup code/retry.sh logs/${n}ab.log $T --name ${n}ab --tgt $t --fset ab > /dev/null 2>&1 < /dev/null &); (nohup code/spfinish.sh ${n}ab $t > /dev/null 2>&1 < /dev/null &); }
go PGr_T85_f20_l814c T85_f20_l814 --anchor mix --fviews robust --bf 128
go PD2_f20_l814 T85_f20_l814 --anchor deg,mix --lamd 5 --dlo 0.6 --dhi 1.0 --fviews thirds
