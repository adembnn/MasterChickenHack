"""Fabrique de faux .pt (format torch.save) SANS torch, pour tester les validateurs. Usage : python3 make_fake_pt.py <dossier>"""
import collections, io, os, pickle, sys, types, zipfile

torch = types.ModuleType("torch"); tu = types.ModuleType("torch._utils")
for n in ("FloatStorage", "BoolStorage", "HalfStorage", "UInt"): setattr(torch, n, type(n, (), {"__module__": "torch"}))
def _rebuild_tensor_v2(*a): pass
_rebuild_tensor_v2.__module__ = "torch._utils"; tu._rebuild_tensor_v2 = _rebuild_tensor_v2
sys.modules["torch"], sys.modules["torch._utils"] = torch, tu

class St:
    def __init__(s, cls, data, numel): s.cls, s.data, s.numel = cls, data, numel
class T:
    def __init__(s, st, size): s.st, s.size = st, size
    def __reduce__(s):
        stride, e = [], 1
        for d in reversed(s.size): stride.append(e); e *= d
        return (_rebuild_tensor_v2, (s.st, 0, tuple(s.size), tuple(reversed(stride)), False, collections.OrderedDict()))

class P(pickle.Pickler):
    def __init__(s, f, store): super().__init__(f, protocol=2); s.store = store
    def persistent_id(s, o):
        if isinstance(o, St):
            k = str(len(s.store)); s.store[k] = o.data
            return ("storage", o.cls, k, "cpu", o.numel)

def save(path, obj):
    store, buf = {}, io.BytesIO(); P(buf, store).dump(obj)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("arch/data.pkl", buf.getvalue()); z.writestr("arch/version", "3\n")
        for k, v in store.items(): z.writestr(f"arch/data/{k}", v)

def ten(cls, data, size, numel=None):
    n = 1
    for s in size: n *= s
    return T(St(getattr(torch, cls), data, numel or n), size)

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from validate_t3 import expected_spec
    spec = expected_spec()
    good = collections.OrderedDict((k, ten("FloatStorage", bytes(4 * eval("*".join(map(str, v)))), v)) for k, v in spec.items())
    save(f"{out}/t3_good.pt", good)
    bad = collections.OrderedDict(good); k0 = "blocks.3.mlp.fc1.bias"
    bad[k0] = ten("FloatStorage", b"\x00\x00\xc0\x7f" + bytes(4 * 3071), (3072,)); save(f"{out}/t3_nan.pt", bad)
    inf = collections.OrderedDict(good); inf[k0] = ten("FloatStorage", b"\x00\x00\x80\xff" + bytes(4 * 3071), (3072,)); save(f"{out}/t3_inf.pt", inf)
    h = collections.OrderedDict(good); h["head.bias"] = ten("HalfStorage", bytes(200), (100,)); save(f"{out}/t3_half.pt", h)
    pre = collections.OrderedDict(("module." + k, v) for k, v in good.items()); save(f"{out}/t3_prefix.pt", pre)
    nok = collections.OrderedDict(good); del nok["head.weight"]; save(f"{out}/t3_missing.pt", nok)
    names = [f"img_{i:04d}" for i in range(1020)]
    save(f"{out}/t1_good.pt", {"image_names": names, "masks": ten("BoolStorage", bytes([i % 2 for i in range(1020 * 16384)]), (1020, 128, 128))})
    save(f"{out}/t1_dup.pt", {"image_names": names[:-1] + [names[0]], "masks": ten("BoolStorage", bytes(1020 * 16384), (1020, 128, 128))})
    save(f"{out}/t1_val2.pt", {"image_names": names, "masks": ten("BoolStorage", b"\x02" + bytes(1020 * 16384 - 1), (1020, 128, 128))})
    save(f"{out}/t1_short.pt", {"image_names": names[:1019], "masks": ten("BoolStorage", bytes(1019 * 16384), (1019, 128, 128))})
    open(f"{out}/test_names.txt", "w").write("\n".join(names))
    hdr = "endpoint,family,model\n"
    gl = [f"ep-{i:02d},{'AB'[i%2]},{'AB'[i%2]}{i%3}" for i in range(1, 34)]
    open(f"{out}/t2_good.csv", "w").write(hdr + "\n".join(gl) + "\n")
    open(f"{out}/t2_missing.csv", "w").write(hdr + "\n".join(gl[:-1]) + "\n")
    open(f"{out}/t2_conflict.csv", "w").write(hdr + "\n".join(gl[:-2] + ["ep-32,A,B0", "ep-33,B,B0"]) + "\n")
    open(f"{out}/t2_ref.csv", "w").write(hdr + "\n".join(gl + ["ref-01,P,S"]) + "\n")
    open(f"{out}/t2_comma.csv", "w").write(hdr + "\n".join(gl[:-1] + ["ep-33,A,A,1"]) + "\n")
