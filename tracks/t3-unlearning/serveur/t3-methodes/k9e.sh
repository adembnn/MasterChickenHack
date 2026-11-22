for n in PGseed1 PGB2 PGB3 PGB6 PGL2 PGlam6 PGs3k PGlr5 PGwcev2 PGwcev3; do for p in $(pgrep -u adem -f "finish.sh ${n}_f70_l814c"); do pkill -P $p; kill $p; done; done; echo nettoye
