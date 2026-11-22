# usage : pack.py out.sh "commande finale" fichier1 fichier2 ...  -> script bash qui recrée les fichiers dans code/ sur Jülich puis lance la commande
import sys, os
out, cmd, files = sys.argv[1], sys.argv[2], sys.argv[3:]
D = '/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3-nonmembres'
s = f'jutil env activate -p training2666 2>/dev/null\ncd {D}/code || exit 1\n'
for f in files:
    src, dst = (f.split('=>') + [None])[:2]; dst = dst or os.path.basename(src)
    body = open(src, encoding='utf-8').read().replace('\r\n', '\n')
    if os.path.dirname(dst): s += f'mkdir -p {os.path.dirname(dst)}\n'
    s += f"cat > {dst} <<'EOF_NM_PACK'\n{body}\nEOF_NM_PACK\n"
s += cmd + '\n'
open(out, 'w', encoding='utf-8', newline='\n').write(s)
