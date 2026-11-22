# T3 deep hunt — Heretic / MPOA / OBLITERATUS vs ViT unlearning, RAZOR, SCOPE, WIN-U, Spectral Unlearning
**22 November 2026, ~05:15 Europe/Paris. Read-only research / action handoff.**

## Ground truth from EAC issue #5 (DO NOT CLAIM A NEW SCORE)

- Team official T3 best: **0.4107**, checkpoint **PG_RDq_70_l915c_s1** (submission 2827). Independent seed gives **0.4042** (submission 2997): seed noise approx **0.0065**; maker-checker now interprets any new public score **below ~0.418** as probably noise. ([Adem's relay and verifier](https://github.com/adembnn/EAC/issues/5)).
- Same base: forget-exact CE mean/std **5.087/4.616**, weak copies **4.384/4.233**. Output distribution T=2 of 8 independent OOF teachers already helped; 16/24 are in flight. Additional inversion retain anchor scored **0.4066** vs 0.4107, so retention updates haven't worked.
- New **CPv3_RDq_70_l612** (team, **NOT scored yet**) raises copy/exact CE mean ratio **0.86→0.94**, BUT changes exact std **4.62→4.29** and proxy-mix KL **0.300→0.395**. Do NOT equate ratio with EAC score. Teammate already queued it **5:28**; no second submission or rerun.
- New temperature-1.5 redistill targets are **already submitted/queued** by the official relay (one test at 5:18). No duplicate.
- EAC scorer requires **strict original timm ViT-B/16 architecture / same weight keys, float32, 100 head classes**; a runtime hook, extra gate, prompt-based technique or extra classifier is **NOT submit-valid**.

## Finding A: Heretic / MPOA / OBLITERATUS (real Reddit/AI hacking communities)

- [Heretic 1.2 Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1r4n3as/heretic_12_released_70_lower_vram_usage_with/) discusses **magnitude-preserving orthogonal abliteration / biprojection (MPOA)**, with layer-specific parameter search and a KL-to-original capacity constraint. [Another Reddit experiment](https://www.reddit.com/r/LocalLLaMA/comments/1ujktg5/normpreserving_abliteration_on_qwen3635ba3b_0/) reports **multi-direction ablation damaged ability benchmarks**, while a better sampled contrastive dataset yielded cleaner one-direction selectivity. These are **LLM-refusal** experiments, NOT classification-unlearning benchmark results.
- [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) implements layer selection, direction extraction (PCA/SVD/mean-diff), norm-preserving edits and **output KL co-optimization**; IMPORTANT it describes **opt-in anonymous telemetry**. Do not upload EAC data, model, host fingerprints, or agent secrets to public Spaces or enable telemetry.
- **Team already claims node10** for **diff-in-means + norm-preserving biprojection on current RDq** ([GPU bus #2](https://github.com/adembnn/EAC/issues/2#issuecomment-7073459384)). Therefore **DO NOT DUPLICATE THIS**. Ask n10 owner for exact-vs-copy CE m/s, class-conditional T2 teacher KL, control gate, unchanged checkpoint schema and random-direction placebo before suggesting any official submission.
- **Causal caveat:** abliteration suppresses a behavior, not necessarily the influence of a training example. EAC rewards matching the *distribution of a retrained gold model*; mere drop in class confidence can worsen the per-example KL. Random multi-direction ablations are a poor bet.

## Finding B (strongest novel method for a 1-training-run trial): Spectral Unlearning (9 November 2026, arXiv:2611.11902)

- **Primary source:** [Spectral Unlearning, arXiv](https://arxiv.org/abs/2611.11902). [Author code](https://github.com/unlearn-ml/spectral-unlearning) (verified readable), Linux/CUDA, but current official recipe is **Llama-3.2 unlearning**, NOT a ready-to-run timm ViT plug-in.
- Instead of changing target RDq, change only optimizer geometry for **2-D ViT attention/MLP weight matrices**, following the spectral-gradient principle. For gradient G=UΣVᵀ, a spectral direction is U Vᵀ (paper uses SOAP/Muon and preconditioning as scalable approximations). Preserve **the same actual RDq teacher targets, 70% selected images, seed, train/val split, weight of mixup/retention anchor and total steps**. Leave layer norms, head and all vectors on AdamW initially; change only matrix updates for last 4 blocks to isolate the factor.
- **Why it might work:** the biggest remaining EAC error is **per-sample shape of the non-target T=2 distribution**, not retain. A different matrix-structured update can reach a different functional solution at the same CE m/s; paper evaluated distributional alignment/JSD to retrained reference and saw a benefit in an LLM setting. **It cannot fix a biased pseudo-gold teacher.**
- **Gate before spending H200:** small **200–400-step paired seed** pilot vs AdamW, report gradient finite/step norms, freeze all nonmatrix layers. DO NOT advance to full 1500-step PG if exact CE m/s cannot be recalibrated near **5.09/4.62**, or teacher heldout per-image T2 KL is worse, or retain control drops. Only one full run if 2 independent seeds show improvements. A simple optimizer swap with unchanged learning rate is **NOT controlled** because the effective step norm changes: match first-step Frobenius norm / effective loss trajectory and count GPU-hours. Then checker and relay only.
- This is **novel vs n10**: it is an optimizer/gradient-direction adaptation during **RDq learning**, not a post-hoc refusal-direction removal or averaging of checkpoint weights.

## Finding C (best read-only diagnostic before n10 abliteration): SCOPE (Aug 2026)

- [SCOPE arXiv 2608.02058](https://arxiv.org/abs/2608.02058) proves a **fixed projector** cannot erase class-discriminant directions without retention loss when those directions overlap the retained classifier readout. It introduces an **input-conditioned gate** that is much more selective; the original work tests *class* forgetting, not 3500 instance-wise deletions.
- EAC demands **same ViT architecture**: adding SCOPE's dynamic inference-time gate is **invalid** as a submission. Also forget features from 45 pseudo-classes reveal primarily **class signal**, not unique memorization of individual images; don't treat those directions as instance-specific.
- New [t3_scope_overlap_diagnostic.py](../tracks/t3-unlearning/experiments/t3_scope_overlap_diagnostic.py): **read-only** class-centroid SVD on original penultimate ViT features, principal-angle / head-readout energy, plus optional *unlabeled proxy retain* effect and random projector controls. Inputs: forget_features.npy [3500,768] from original timm ViT forward_head(pre_logits=True), head_weight.npy [100,768], optional head_bias.npy [100], retain_features.npy [M,768].
- Suggested command on an already connected/reserved agent machine (CPU if caches exist):

        python tracks/t3-unlearning/experiments/t3_scope_overlap_diagnostic.py --smoke
        python tracks/t3-unlearning/experiments/t3_scope_overlap_diagnostic.py \
          --forget-features original_forget_feats.npy --head original_head_weight.npy \
          --bias original_head_bias.npy --retain-features independent_proxy_feats.npy \
          --ranks 1,2,4,8 --out scope_overlap.json

- **Numerical primitive verified on synthetic arrays**, where cosine²≈1 and 91.8% of the head's readout energy overlaps class axes, producing large simulated JS. This is **expected** on class-structured synthetic features and **NOT an EAC measurement**. The committed script is not yet run on the real EAC caches.
- **Gate:** if random direction controls are much less damaging than class-prototype projectors on independent proxies, warn n10 against fixed global projection; prioritize the already strong targeted RDq training. Low damage does not itself predict gold KL, so no 'GO' without exact/copy EAC surrogate evaluation.

## Finding D (second-best prospective new trial): RAZOR (CVPR 2026 Findings)

- [RAZOR arXiv 2603.14819](https://arxiv.org/abs/2603.14819) explicitly tests **multi-layer and attention-head editing for vision/CLIP/diffusion**, much closer to EAC ViT than generic LLM refusal.
- Their *ranking* is phi_l = (||g_forget,l|| / (||theta_l||+eps)) * (1-cos(g_forget,l,g_retain,l))^alpha, and they then update high-rank layers selectively. The official method presumes **retain data and labels/losses**, not available in EAC. The paper's cited code URL **could not be fetched (GitHub 404 when checked)**; do not claim a turnkey implementation.
- **Adaptation**: compute g_forget using KL-to-RDq teacher on a stratified sample of 256 forget images, g_proxy using KL-to-orig on 256 truly independent proxy nonmembers (or mixups, but those are not true retain). Measure the ranking of blocks 6–11 and heads. If it chooses essentially 8–11 (current PG), **STOP**. If it consistently picks a different *small* set on two disjoint batches, test 1 matched-step matched-target seed of those blocks vs conventional last-four blocks. More data required: proxy gradient quality is a serious source of bias.
- This is distinct from t3-poids' old Fisher exact-vs-augmented result, which found near-perfect correlation (~0.996) and did **not compare teacher-target forget gradients vs independent proxy retain gradients**. However **t3-poids already measured block 6–11 as more surgical** than all layers; not claiming a totally new block-unlearning concept.

## Findings E/F: WIN-U / SLUG — informative but low immediate priority

- [WIN-U 2604.13438](https://arxiv.org/abs/2604.13438) uses generalized Gauss–Newton + Woodbury to approximate the leave-forget-out retrained optimum **without retain data**, and can in principle adjust the target of forget outputs. But correct labels/Hessian at trained optimum are unavailable and rank-3500 curvature can be expensive; this is **not a drop-in for PG**. First test a small *last-layer analytical* approximation with pseudo-labels only if spectral trial fails and teacher caches available; EAC head-only retraining already recovered 100% of memorized forget images (#5 7068757075), so a simple reset is NO-GO.
- [SLUG ICML2025](https://proceedings.mlr.press/v267/cai25f.html), [code](https://github.com/CSIPlab/SLUG), selects one layer and one gradient with low retain alignment. Different ViT task and needs retain data. **Do not repeat head-only reset or all-layer NPO**.

## Rejected “hacker bypass” ideas
- Generic jailbreaking suffixes, refusal-token probabilities, chat-sampler changes: **there is no text generation component in EAC ViT**.
- Installing/running Heretic or OBLITERATUS on the image ViT *as is*: unsupported architecture and loss, risks invalid checkpoint; their communities' high ASR does not predict T3.
- Submitting a runtime SCOPE feature gate or a model with an added adapter: server requires exact timm state_dict.
- Blind multiple-direction ablations, stronger CE, heavier retain anchors, 16/24 redistill duplicated: already likely harmful or in progress.

## Ranked handoff and source integrity
1. **Now:** wait on currently queued official RDqp (5:08), q15 (5:18), CPv3 (5:28), and RDq16; don't steal queue. n10 owner can perform SCOPE feature-readout overlap diagnostic **before** extending its existing abliteration tests; report no edits if conflict.
2. **If no public >~0.418:** one matched-seed pilot of spectral-gradient optimizer on the **existing RDq** targets (only if a GPU slot and an agent agrees; NOT run by this assistant). Detect lower heldout per-image teacher KL with m/s and gate fixed before full training.
3. **If spectral pilots fail:** one RAZOR-inspired gradient-ratio readout on existing forget and independent proxy batch; train at most one selected-block variant if disjoint-batch rank reproducible.
4. **Other ideas:** WIN-U targeted head curvature and SCOPE distillation are research only for now; avoid saturating the relay with unvalidated files.

**Status**: I accessed the source papers/repo READMEs and EAC issue comments through GitHub, committed the read-only diagnostic and this report. No H200 run, no EAC endpoint call, no official T3 submission; no secret in branch. This is **evidence-based test triage**, not a new leaderboard improvement. 
