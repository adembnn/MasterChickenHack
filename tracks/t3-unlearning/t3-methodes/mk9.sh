#!/bin/bash
# cibles PGh / PGo / PGho a 50, 60, 70 % (loss 8-14) ; usage : mk9.sh PYTHON
cd ~/eac/t3/t3-methodes; M="$1 code/mk_tgt.py"; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
for f in 50 60 70; do B="--pool $P85 --f 0.$f --llo 8 --lhi 14"
  $M H$f $B --softn 0 | tail -n 1 | cut -c1-5,330-440; $M O$f $B --wsrc orig > /dev/null; $M HO$f $B --softn 0 --wsrc orig > /dev/null; done; ls cache | grep -c -E "tgt_(H|O|HO)[567]0"
