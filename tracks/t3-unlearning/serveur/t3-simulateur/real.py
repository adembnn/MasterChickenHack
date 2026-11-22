"""Applique au VRAI probleme une recette notee dans le simulateur : vrai model.pt + vraies images a oublier (aucune etiquette, aucune autre image).
usage : real.py CONFIGS.json   -> real/cand_<nom>.pt (state dict float32) + real/diag_<nom>.json
Les diagnostics sont calcules contre la pseudo-etiquette (prediction d'orig), la seule disponible ; ce ne sont PAS des scores."""
from methods import *


def diag(B, m, name, c):
    m.eval(); N = B.N; yh = B.yhat; out = {'name': name, 'cfg': {k: v for k, v in c.items() if not k.startswith('_')}}
    cnt = torch.bincount(yh, minlength=NC); full = (cnt[yh] >= c.get('full_thr', 0.97) * c.get('ctot', 130)).cpu()
    g = gen(4242)
    def stats(z, tag):
        z = z.float(); ce = F.cross_entropy(z, yh, reduction='none').cpu(); wr = (z.argmax(1) != yh).float().cpu(); p2 = F.softmax(z / 2, 1).max(1).values.cpu()
        out[tag] = {'loss_moy': ce.mean().item(), 'loss_sd': ce.std().item(), 'classes_changees': wr.mean().item(), 'maxprob_T2': p2.mean().item(),
                    'entieres': {'loss_moy': ce[full].mean().item(), 'loss_sd': ce[full].std().item(), 'classes_changees': wr[full].mean().item(), 'maxprob_T2': p2[full].mean().item()},
                    'autres': {'loss_moy': ce[~full].mean().item(), 'loss_sd': ce[~full].std().item(), 'classes_changees': wr[~full].mean().item(), 'maxprob_T2': p2[~full].mean().item()}}
        return z
    with torch.no_grad(), ac():
        zs = stats(torch.cat([m(norm(to01(B.fg[i:i + 250]))) for i in range(0, N, 250)]), 'exact')
        for t in range(4): stats(torch.cat([m(norm(altered(to01(B.fg[i:i + 250]), t, g))) for i in range(0, N, 250)]), f'copies : {ALT_NAMES[t]}')
        za = torch.cat([top(m, B.Tm[i:i + 500]).float() for i in range(0, len(B.Tm), 500)])
    kla = kl(F.softmax(B.Zm / 2, 1), F.log_softmax(za / 2, 1)).cpu(); agree = (za.argmax(1) == B.Zm.argmax(1)).float().cpu()
    out['derive'] = {'melanges inter-classes : KL(orig||sub) T=2': kla[:B.nmix].mean().item(), 'accord argmax': agree[:B.nmix].mean().item(),
                     'melanges intra-classe : KL(orig||sub) T=2': kla[B.nmix:].mean().item(), 'accord argmax ': agree[B.nmix:].mean().item()}
    # vers quelle classe partent les classes entieres
    pr = zs.argmax(1).cpu(); yc = yh.cpu(); out['bascules'] = {int(c): {int(k): int(v) for k, v in zip(*np.unique(pr[yc == c].numpy(), return_counts=True))} for c in yc[full].unique()}
    sd = {k: v.detach().cpu().float().contiguous() for k, v in m.state_dict().items()}; ref = build(); miss = ref.load_state_dict(sd, strict=True)
    out['fichier'] = {'strict': str(miss), 'float32': all(v.dtype == torch.float32 for v in sd.values() if v.is_floating_point()), 'nan_inf': bool(any((~torch.isfinite(v)).any() for v in sd.values() if v.is_floating_point())),
                      'logits_fp16_finis': bool(torch.isfinite(zs).all())}
    path = f'{B.rdir}/cand_{name}.pt'; torch.save(sd, path); out['fichier']['Mo'] = os.path.getsize(path) / 1e6; out['chemin'] = path
    json.dump(out, open(f'{B.rdir}/diag_{name}.json', 'w'), indent=1)
    e = out['exact']; print(f"{name}: exact loss {e['loss_moy']:.3f} sd {e['loss_sd']:.3f} classes changees {e['classes_changees']:.3f} | entieres : loss {e['entieres']['loss_moy']:.2f} changees {e['entieres']['classes_changees']:.3f} maxprob T2 {e['entieres']['maxprob_T2']:.3f}"
          f" | autres : loss {e['autres']['loss_moy']:.4f} changees {e['autres']['classes_changees']:.4f} | derive KL inter {out['derive']['melanges inter-classes : KL(orig||sub) T=2']:.4f} intra {out['derive']['melanges intra-classe : KL(orig||sub) T=2']:.4f} | {path}", flush=True)
    for t in range(4): a = out[f'copies : {ALT_NAMES[t]}']; print(f"   copies {ALT_NAMES[t]:22s}: loss {a['loss_moy']:.3f} sd {a['loss_sd']:.3f} changees {a['classes_changees']:.3f} (entieres {a['entieres']['classes_changees']:.3f}, autres {a['autres']['classes_changees']:.4f})", flush=True)


if __name__ == '__main__':
    warm(); B = Bench('REAL', None, [], V=int(os.environ.get('BENCH_V', 4)), M=int(os.environ.get('BENCH_M', 6000)))
    cnt = torch.bincount(B.yhat, minlength=NC); print('classes entieres (>= 129 images a oublier) :', [int(c) for c in (cnt >= 129).nonzero().flatten()], 'images :', int(cnt[cnt >= 129].sum()), flush=True)
    for c in json.load(open(sys.argv[1])):
        for essai in range(6):
            try:
                t0 = time.time(); m, _, info = B.fit(c); print(c['name'], '|', info, f'| {time.time() - t0:.0f}s', flush=True); diag(B, m, c['name'].replace(' ', '_'), c); del m; break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' not in str(e) and 'ALLOC_FAILED' not in str(e): raise
                print('memoire pleine, nouvel essai dans 45 s', flush=True); time.sleep(45)
