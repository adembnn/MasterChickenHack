#!/bin/bash
# arrête x9st (4e réplique à tige passe-haut, à 1000 pas) : t1-assemblage dit que les points restants sont sur les pleines ; la carte va aux modèles proches de la fin et au spécialiste des pleines
pkill -f "16000 x9st"; pkill -f "tag g_x9st"; sleep 3; ps -eo etimes,args | grep "reel1[12].py train" | grep -v grep | awk '{for(i=1;i<=NF;i++) if($i=="--tag") print $1, $(i+1)}'
