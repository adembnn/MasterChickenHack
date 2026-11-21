"""A lancer SUR JURECA (torch + timm). NON TESTE ici (pas de torch sur le Mac). Copie fidele de ce que fait le serveur :
chargement strict, eval, autocast fp16, 3500 images oubli. Usage : python selfcheck_t3.py submission.pt
(task_template.py doit etre dans le dossier ; model.pt et forget_images.npz aussi)."""
import sys, torch
import task_template as tt
sub = torch.load(sys.argv[1], map_location="cpu", weights_only=True)          # comme un serveur prudent
assert all(v.dtype == torch.float32 for v in sub.values()), "dtype != float32"
assert all(torch.isfinite(v).all() for v in sub.values()), "NaN/Inf"
m = tt.build_model(); m.load_state_dict(sub, strict=True); m = m.to(tt.DEVICE).eval()
orig = tt.load_model(); imgs = tt.load_forget_images()
ls, lo = tt.predict(m, imgs), tt.predict(orig, imgs)                            # fp16 autocast comme le serveur
assert torch.isfinite(ls).all(), "logits non finis sous fp16 autocast"
pu, po = (ls / 2).log_softmax(1), (lo / 2).softmax(1)
kl = (po * (po.clamp_min(1e-12).log() - pu)).sum(1).mean().item()
print(f"OK. top1 identique a orig sur oubli: {(ls.argmax(1)==lo.argmax(1)).float().mean():.3f}  KL(orig||sub,T=2)={kl:.4f}")
print("rappel : porte = 70 % d'accuracy sur une validation cachee ; mesurez sur un proxy de retention AVANT de soumettre.")
