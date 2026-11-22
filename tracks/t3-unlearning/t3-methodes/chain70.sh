#!/bin/bash
cd ~/eac/t3/t3-methodes
until [ $(cat logs/PGh_f50_l814c.log logs/PGh_f50_l814cab.log logs/PGh_f60_l814c.log logs/PGh_f60_l814cab.log 2>/dev/null | grep -c "^saved") -ge 4 ]; do sleep 10; done
code/wave.sh "PGho_f70_l814c HO70 M" "PGh_f70_l814c H70 M"
