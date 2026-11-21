"""Lecteur minimal de fichiers torch.save (.pt/.pth) en Python standard (sans torch ni numpy).
Refuse tout objet hors liste blanche (comme torch.load(weights_only=True) cote serveur). Python >= 3.8."""
import collections, pickle, zipfile

STORAGES = {"FloatStorage": ("float32", 4), "HalfStorage": ("float16", 2), "BFloat16Storage": ("bfloat16", 2),
            "DoubleStorage": ("float64", 8), "LongStorage": ("int64", 8), "IntStorage": ("int32", 4),
            "ShortStorage": ("int16", 2), "CharStorage": ("int8", 1), "ByteStorage": ("uint8", 1),
            "BoolStorage": ("bool", 1)}


class StorageType:
    def __init__(self, name): self.name = name


class Storage:
    def __init__(self, kind, key, numel): self.kind, self.key, self.numel = kind, key, numel


class Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)
        self.dtype, self.itemsize = STORAGES[storage.kind]

    @property
    def numel(self):
        n = 1
        for s in self.size: n *= s
        return n

    def contiguous(self):
        exp, st = 1, []
        for s in reversed(self.size): st.append(exp); exp *= max(s, 1)
        return all(a == b or d == 1 for a, b, d in zip(self.stride, tuple(reversed(st)), self.size))


def _rebuild(storage, offset, size, stride, *rest):
    return Tensor(storage, offset, size, stride)


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"): return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2": return _rebuild
        if module == "torch._utils" and name == "_rebuild_parameter": return lambda data, *a: data
        if module == "torch" and name in STORAGES: return StorageType(name)
        raise pickle.UnpicklingError(f"objet interdit {module}.{name} (weights_only le refuserait)")

    def persistent_load(self, pid):
        if pid[0] != "storage": raise pickle.UnpicklingError(f"persistent id inconnu {pid[0]}")
        return Storage(pid[1].name, str(pid[2]), int(pid[4]))


class PtFile:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        pk = [n for n in self.zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
        if len(pk) != 1: raise ValueError(f"data.pkl introuvable ou multiple: {pk}")
        self.prefix = pk[0][: -len("data.pkl")]
        self.obj = _U(__import__("io").BytesIO(self.zf.read(pk[0]))).load()

    def storage_bytes(self, t):
        return self.zf.read(f"{self.prefix}data/{t.storage.key}")


def has_nonfinite_f32(data: bytes) -> bool:
    """True si un float32 little-endian est NaN ou Inf (exposant = 255). Rapide : tranches + entiers longs."""
    b3, b2 = data[3::4], data[2::4]
    t7f = bytes(1 if (i & 0x7F) == 0x7F else 0 for i in range(256))
    t80 = bytes(1 if i >= 0x80 else 0 for i in range(256))
    return (int.from_bytes(b3.translate(t7f), "little") & int.from_bytes(b2.translate(t80), "little")) != 0
