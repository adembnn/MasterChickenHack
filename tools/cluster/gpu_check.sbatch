#!/bin/bash
# Vérifie qu'un job tourne bien sur un nœud de calcul avec un GPU A100 alloué par Slurm,
# et si ce nœud a accès à internet. Durée : moins de 2 minutes, 1 GPU.
# Usage (depuis le nœud de login) : sbatch gpu_check.sbatch ; puis cat gpucheck-<jobid>.out
#SBATCH --job-name=gpucheck
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

echo "== OÙ TOURNE LE JOB =="
echo "hostname=$(hostname)  SLURM_JOB_ID=$SLURM_JOB_ID  SLURMD_NODENAME=$SLURMD_NODENAME"
echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION  reservation=cispahack  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo
echo "== GPU VUS PAR LE JOB =="
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv

PY=/p/scratch/training2666/destors1/MasterChickenHack/unlearning/.venv/bin/python
echo
echo "== PYTORCH =="
$PY - <<'EOF'
import time, torch
print("torch", torch.__version__, "| cuda dispo:", torch.cuda.is_available(), "| nb GPU:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 0:", torch.cuda.get_device_name(0), "| mémoire totale (Go):", round(torch.cuda.get_device_properties(0).total_memory/1e9,1))
    n=8192
    a=torch.randn(n,n,device="cuda",dtype=torch.float16); b=torch.randn(n,n,device="cuda",dtype=torch.float16)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(20): c=a@b
    torch.cuda.synchronize(); dt=time.time()-t
    print("20 produits matriciels 8192x8192 fp16 sur GPU: %.2f s (~%.0f TFLOPS)" % (dt, 20*2*n**3/dt/1e12))
    x=torch.randn(2048,2048); t=time.time()
    for _ in range(3): y=x@x
    print("référence CPU (3 produits 2048x2048): %.2f s" % (time.time()-t))
else:
    print("PAS DE GPU VU PAR TORCH")
EOF

echo
echo "== INTERNET DEPUIS LE NŒUD DE CALCUL (timeout 8 s) =="
for u in https://huggingface.co https://pypi.org https://github.com http://34.147.229.55/leaderboard_page; do
  code=$(curl -s -m 8 -o /dev/null -w "%{http_code}" "$u" 2>/dev/null)
  echo "$u -> HTTP ${code:-aucune réponse}"
done
echo "== fin =="
