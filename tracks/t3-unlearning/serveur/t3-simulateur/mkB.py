import common; common.dev = "cpu"; common.make_world("B", 130, 0)
import numpy as np; w = np.load(common.DATA + "/world_B.npz"); Y = np.load(common.DATA + "/y.npy")
r = np.bincount(Y[w["retain"]], minlength=100); print("images restantes par classe pour le gold : min", r.min(), "; classes a 0 :", int((r == 0).sum()), "; classes < 30 :", int((r < 30).sum()))
