# annule l'envoi f70 en attente (processus sub.sh et son sleep)
for p in $(pgrep -u adem -f "sub.sh .*PG_T85_f70_l814c"); do pkill -P $p; kill $p; done
sleep 1; pgrep -u adem -fa "sub.sh" | cut -c1-120; echo "--- journal f70:"; tail -3 ~/eac/t3/orchestrateur/sub_PG_T85_f70_l814c.log | cut -c1-160
echo "--- G10:"; grep -rl "G10" ~/eac/t3/*/code ~/eac/t3/*/*.py ~/eac/t3/orchestrateur 2>/dev/null | head; ls ~/eac/t3/*/candidates 2>/dev/null | grep -i "G10"
