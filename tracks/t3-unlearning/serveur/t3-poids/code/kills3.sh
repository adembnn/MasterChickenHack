for p in $(ps -u adem -o pid,args | grep "code/sum3.sh" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 1
for p in $(ps -u adem -o pid,args | grep "dash.py --fset all --rset none --tgt T85 --tag all_ --combo L814c" | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 2; rm -f ~/eac/t3/t3-poids/candidates/L814c_NEG*.pt; ps -u adem -o pid,etimes,args | grep "t3-poids\|unl.py\|dash.py" | grep -v grep | cut -c1-110
