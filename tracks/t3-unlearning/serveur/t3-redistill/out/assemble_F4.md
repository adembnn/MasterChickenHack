F4_f0_logits.npz images ecartees 717
F4_f1_logits.npz images ecartees 707
F4_f2_logits.npz images ecartees 698
F4_f3_logits.npz images ecartees 692
F4_f4_logits.npz images ecartees 686
redistill.npy (3500, 100) float32 | precision vs classe d orig 0.633 | loss moy 1.05 sd 1.32 | loss des ratees 2.38 | max-prob T1 0.728 T2 0.457
  classe fausse = 2e choix d orig 0.513 | vers une des 55 classes absentes 0.175
  copie jpeg50 : ratees 0.390 loss 1.13/1.39 ; meme verdict (juste/faux) que sur l exacte 0.921 ; meme classe predite 0.895
  copie rot15 : ratees 0.386 loss 1.13/1.41 ; meme verdict (juste/faux) que sur l exacte 0.756 ; meme classe predite 0.685
  copie cropflip : ratees 0.376 loss 1.08/1.36 ; meme verdict (juste/faux) que sur l exacte 0.779 ; meme classe predite 0.709
  copie mild : ratees 0.391 loss 1.16/1.42 ; meme verdict (juste/faux) que sur l exacte 0.797 ; meme classe predite 0.730
  taux d erreur par classe : min 0.13 mediane 0.33 max 0.71 ; correlation avec l effectif de la classe 0.270

| selecteur | sdxl_i2i_70 : AUC / meme classe fausse | sdxl_i2i_85 : AUC / meme classe fausse | sdxl_ipa_60 : AUC / meme classe fausse |
|---|---|---|---|
| redistill hors pli (4 blocs) | 0.648 / 0.442 (n=95) | 0.580 / 0.392 (n=158) | 0.553 / 0.391 (n=128) |
| oofA_b8 (t3-poids, sonde lineaire hors pli) | 0.531 / 0.194 (n=72) | 0.507 / 0.200 (n=135) | 0.493 / 0.214 (n=112) |
| tgt_T85_f30_l814 (t3-methodes) | 0.653 / 0.386 (n=83) | 0.624 / 0.356 (n=149) | 0.560 / 0.300 (n=110) |
| tgt_T85_f20_l814 (t3-methodes) | 0.657 / 0.353 (n=68) | 0.626 / 0.324 (n=111) | 0.563 / 0.296 (n=81) |
| tgt_S30_oof (t3-methodes) | 0.591 / 0.141 (n=64) | 0.545 / 0.195 (n=118) | 0.516 / 0.240 (n=100) |
| tgt_S30_err (t3-methodes) | 0.653 / 0.386 (n=83) | 0.624 / 0.356 (n=149) | 0.560 / 0.300 (n=110) |
| bruit sur les poids s0.6 (sans entrainement) | 0.612 / 0.000 (n=1) | 0.573 / 0.000 (n=1) | 0.552 / 0.500 (n=2) |
| bruit sur les poids s0.9 (sans entrainement) | 0.596 / 0.400 (n=20) | 0.547 / 0.348 (n=23) | 0.538 / 0.429 (n=21) |
| bruit sur les poids s1.2 (sans entrainement) | 0.536 / 0.312 (n=96) | 0.525 / 0.210 (n=186) | 0.530 / 0.261 (n=157) |
| bruit sur les poids s1.5 (sans entrainement) | 0.483 / 0.152 (n=151) | 0.505 / 0.117 (n=332) | 0.521 / 0.095 (n=284) |
| marge d orig sur l exacte (temoin) | 0.584 / - | 0.579 / - | 0.561 / - |
Lecture : AUC = probabilite qu une image dont la regeneration trompe orig recoive du selecteur une loss plus haute qu une image dont la regeneration ne le trompe pas (0,5 = hasard). Part des regenerations qui trompent orig : sdxl_i2i_70 0.204 ; sdxl_i2i_85 0.432 ; sdxl_ipa_60 0.376
