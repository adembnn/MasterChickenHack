import numpy as np, hashlib
a = np.load('data/alt_A.npy', mmap_mode='r')
for t in range(4):
    x = np.ascontiguousarray(a[t, :200]); print(t, hashlib.md5(x.tobytes()).hexdigest()[:10], 'moyenne %.5f' % x.mean(), 'img0 %.4f img199 %.4f' % (x[0].mean(), x[199].mean()), 'coin', x[7, 100, 100].tolist())
