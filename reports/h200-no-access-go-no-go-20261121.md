# H200 availability + submit gates (21 Nov 2026, ~23:56 Europe/Paris)

**Status: BLOCKED — no authorized machine reachable from this ChatGPT session. No GPU/contest job/submit/API calls performed.**

## Attempted checks, and why they do not prove availability
- `https://h200.ami.exchange/` could not be resolved in this runtime (DNS error). The public leaderboard IP did not connect.
- Direct TCP 22 attempts to the **four documented public nodes** `66.201.7.7` (1), `66.201.4.154` (2), `66.201.6.94` (3), `204.12.188.1` (6) all returned **Connection refused** from this sandbox. This may be a network sandbox restriction; **do not infer that the VM itself is down or idle.**
- SSH private keys `/home/boudda/sshkey/vultr_ed25519`, `/clients/ami/.ssh/eac_nebius_h200` aren't mounted here. GitHub contains **paths and public host fingerprints only**, not keys. No attempt to fake access.
- Latest issue [#2](https://github.com/adembnn/EAC/issues/2) allocates n1 to T2 null-model readers and contains recent activity on 1–6. Only `nvidia-smi` **plus** current process-owner coordination can confirm that a node is free. A stale dashboard green tile is insufficient.

## What a connected agent must do before starting any job
1. On the user's laptop or Hermes NemoClaw **where the private key actually resides**, connect using the documented per-node SSH alias in `docs/acces-h200-nebius*.md`. Verify the **host fingerprint** before accepting it.
2. Run `nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader` and `nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader`; inspect `ps -eo user,pid,cmd --sort=-rss | head`.
3. Read fresh [issue #2](https://github.com/adembnn/EAC/issues/2), announce proposed VRAM and duration, **ask Adem/Titouan** when GPU memory is shared. No force-kill and no preemption.
4. Confirm `task_template.py` and `validate_t{1,3}.py` paths (do not assume they are on a new node). Keep results under a fresh own directory. Do not copy secrets to GitHub.

## Current certified public baselines
- **T1**: `submission_candidat_22.pt` → public **0.7695** (submit 1984), reserve **0.7514**. Latest candidate24 `0.7536` reserve (+.0022 vs 22), valid file but **NOT approved by the +.020 improvement gate**. Another agent announced a submit attempt of candidate24, result **not yet confirmed** by issue #3. Must check latest official score before making assumptions.
- **T3**: `PG_T85_f80_l814c` public **0.3179** (submit 1979). f70 = **0.3159**. Fraction sweep plateau.

## Lowest-cost real evaluation if one node becomes available
**Priority A, T3 (zero retraining)**: use [`t3_pg_weight_mix.py`](../tracks/t3-unlearning/experiments/t3_pg_weight_mix.py) to create wB .25/.50/.75 strict float32 mixes between independently validated f70/f80 files (readable checkpoints owned by Adem). Check strict model load, fp16 forward, forget **all 3500 exact and changed** samples, per-sample CE moments, holdout proxy retain T2 KL + gate ≥70%, then get maker-checker. Discard variants whose forgetting collapses or retain falls. The CPU synthetic test of lerp was reported; **actual contest checkpoints remain untested**. This quick ablation has small upside, not a predicted victory.

**Priority B, T3 (GPU training, coordinated)**: Titouan has already launched/waitlisted `all7_cons75` vs `all7_inst75` on h200c; **do NOT duplicate it**. Only if his result is positive and Adem agrees, port his consensus selector into proven PG-f70/f80 recipe, preserving the explicit CE-height term. His +9% m+s and blur-accuracy 0.768→0.841 were at f30, **not** f75 and cannot be assumed to transfer.

**T1**: only consider replacing candidate22 if reserve increase ≥+.020 (target ≥.7714), paired CI95 excludes 0 and Adem's checker issues written approval. A simple StegaStamp +.015 sub-track gain ≈+.005 overall doesn't meet that bar alone. Prioritize PixelSeal small clean-hole signal (paired counterfactual loss PR #34), but this needs a full matched evaluation, not just a smoke test.

## Submit
Only approved candidates get a slot; confirm official latest leaderboard score and coordinate the shared per-team submission cooldown with Adem first. T2 fully delegated to Titouan. Never place the EAC API key in commands, docs, issues, logs or PRs; use the machine's protected environment variable / curl config.

**No H200 availability established, no contest submission performed in this session.**
