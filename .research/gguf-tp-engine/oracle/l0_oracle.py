#!/usr/bin/env python3
"""L0 oracle (class A) for the GGUF-TP format contract.

Reference A: pinned llama.cpp dequantization (ref_a.so, verbatim extraction
from Whamp/llama.cpp @ 0379cf4bf — see ref_a.c header).
Reference B: independent decoders written ONLY from FORMAT-CONTRACT.md.

Pass = 100% bitwise-equal fp32 outputs across random + adversarial corpora
for q8_0, q2_K, iq2_xxs. Writes evidence/l0-report.json; exit 1 on mismatch.

NumPy float32 ops replicate the contract's operation order exactly.
fp16->fp32 via astype is IEEE-exact; NaN payloads (possible only when a
scale field encodes NaN — masked out of the main corpus) are compared
NaN-aware in the dedicated probe.
"""

import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = Path("/home/will/projects/llama.cpp-ds4-study/ggml/src")
EVID = HERE.parent / "evidence"
SEED = 20260817
N_RANDOM_BLOCKS = 10_000

QK_K, QK8_0 = 256, 32
SZ = {"q8_0": 34, "q2_K": 84, "iq2_xxs": 66}
QK = {"q8_0": QK8_0, "q2_K": QK_K, "iq2_xxs": QK_K}
F32 = np.float32


# ---------------- Reference B: independent decoders (from contract) ---------

def dequant_q8_0(raw, tabs):
    """fp16 d @0, int8 qs[32] @2. y[i] = d * qs[i]."""
    n = raw.shape[0]
    d = raw[:, 0:2].copy().view(np.float16).astype(F32).reshape(n)
    qs = raw[:, 2:34].copy().view(np.int8).astype(F32).reshape(n, QK8_0)
    return (d[:, None] * qs).reshape(-1)


def dequant_q2_K(raw, tabs):
    """scales[16]@0, qs[64]@16, fp16 d@80, dmin@82. x = dl*q - ml.
    Two 128-weight chunks; per chunk 4 shift stages (0,2,4,6); each stage
    covers two sub-blocks: first reads qs[qbase..qbase+15], second
    qs[qbase+16..qbase+31]; scale nibbles consumed in the same order."""
    n = raw.shape[0]
    d = raw[:, 80:82].copy().view(np.float16).astype(F32).reshape(n)
    dmin = raw[:, 82:84].copy().view(np.float16).astype(F32).reshape(n)
    scales = raw[:, 0:16]
    qs = raw[:, 16:80]

    out = np.empty((n, QK_K), dtype=F32)
    iscale, qbase = 0, 0
    for chunk in range(2):
        for j in range(4):
            shift = 2 * j
            for half in range(2):
                sc = scales[:, iscale]
                dl = d * (sc & 0xF).astype(F32)
                ml = dmin * (sc >> 4).astype(F32)
                qoff = qbase + 16 * half
                q2 = ((qs[:, qoff:qoff + 16] >> shift) & 3).astype(F32)
                w0 = 128 * chunk + 32 * j + 16 * half
                out[:, w0:w0 + 16] = (dl[:, None] * q2) - ml[:, None]
                iscale += 1
        qbase += 32
    return out.reshape(-1)


def dequant_iq2_xxs(raw, tabs):
    """fp16 d @0, qs[32]@2 = 8 groups of 8 bytes.
    Per group: aux32[0]=bytes 0..3, aux32[1]=bytes 4..7 (LE); aux8 = aux32[0]
    bytes; db = (d * (0.5 + (aux32[1]>>28))) * 0.25;
    grid8 = iq2xxs_grid[aux8[l]] (8 bytes, uint64 LE);
    signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
    y = (db * grid8[j]) * (bit j of signs ? -1 : +1)."""
    grid = tabs["iq2xxs_grid"]          # (256, 8) uint8
    ksigns = tabs["ksigns_iq2xs"]       # (128,) uint8
    kmask = tabs["kmask_iq2xs"]         # (8,) uint8
    n = raw.shape[0]
    d = raw[:, 0:2].copy().view(np.float16).astype(F32).reshape(n)
    qs = raw[:, 2:66].reshape(n, 8, 8)

    u32 = lambda col: (qs[:, :, col].astype(np.uint32)
                       | (qs[:, :, col + 1].astype(np.uint32) << np.uint32(8))
                       | (qs[:, :, col + 2].astype(np.uint32) << np.uint32(16))
                       | (qs[:, :, col + 3].astype(np.uint32) << np.uint32(24)))
    a0, a1 = u32(0), u32(4)

    db = ((d[:, None, None] * (F32(0.5) + (a1 >> np.uint32(28)).astype(F32))[..., None])
          * F32(0.25))                                            # (n, 8, 1)

    out = np.empty((n, 8, 32), dtype=F32)
    for l in range(4):
        g = grid[qs[:, :, l].astype(np.int32)]                     # (n,8,8)
        signs = ksigns[((a1 >> np.uint32(7 * l)) & np.uint32(127)).astype(np.int32)]
        neg = (signs[:, :, None] & kmask[None, None, :]) != 0
        prod = db * g.astype(F32)
        out[:, :, 8 * l:8 * l + 8] = np.where(neg, prod * F32(-1.0), prod * F32(1.0))
    return out.reshape(-1)


DEC = {"q8_0": dequant_q8_0, "q2_K": dequant_q2_K, "iq2_xxs": dequant_iq2_xxs}


# ---------------- corpora ----------------------------------------------------

def finite_fp16(bits):
    return (bits & np.uint16(0x7C00)) != np.uint16(0x7C00)


def gen_random(name, rng, n):
    b = rng.integers(0, 256, size=(n, SZ[name]), dtype=np.uint8)
    fields = {"q8_0": [(0, 2)], "q2_K": [(0, 2), (80, 82), (82, 84)],
              "iq2_xxs": [(0, 2)]}[name]
    for lo, hi in fields:
        db = b[:, lo:hi].view(np.uint16).reshape(-1)
        while True:
            bad = ~finite_fp16(db)
            k = int(bad.sum())
            if k == 0:
                break
            repl = rng.integers(0, 256, size=(k, 2), dtype=np.uint8).view(np.uint16).reshape(-1)
            db[bad] = np.where(finite_fp16(repl), repl, np.uint16(0x3C00))
    return b


def bytes16(v):
    return int(v).to_bytes(2, "little")


def adversarial(name):
    z = lambda: np.zeros((1, SZ[name]), dtype=np.uint8)
    if name == "q8_0":
        return [
            ("qs=-128", None, lambda c: c.__setitem__((slice(None), slice(2, 34)), 0x80)),
            ("qs=127", None, lambda c: c.__setitem__((slice(None), slice(2, 34)), 0x7F)),
            ("qs=alt", None, lambda c: c.__setitem__((slice(None), slice(2, 34)),
                                                      np.tile(np.array([0x80, 0x7F], dtype=np.uint8), 16))),
            ("d=max", bytes16(0x7BFF), lambda c: c.__setitem__((slice(None), slice(2, 34)), 0x7F)),
            ("d=subnorm", bytes16(0x0001), lambda c: c.__setitem__((slice(None), slice(2, 34)), 0x80)),
            ("d=-max", bytes16(0xFBFF), lambda c: c.__setitem__((slice(None), slice(2, 34)), 0x7F)),
        ]
    if name == "q2_K":
        return [
            ("scales=0xFF", None, lambda c: c.__setitem__((slice(None), slice(0, 16)), 0xFF)),
            ("scales=0x0F", None, lambda c: c.__setitem__((slice(None), slice(0, 16)), 0x0F)),
            ("scales=0xF0", None, lambda c: c.__setitem__((slice(None), slice(0, 16)), 0xF0)),
            ("qs+scale-max", bytes16(0x7B00), lambda c: (c.__setitem__((slice(None), slice(0, 16)), 0xFF),
                                                          c.__setitem__((slice(None), slice(16, 80)), 0xFF))),
            ("d=+max", bytes16(0x7BFF), lambda c: c.__setitem__((slice(None), slice(0, 16)), 0xFF)),
            ("dmin=+max", None, lambda c: (c.__setitem__((slice(None), slice(82, 84)), np.frombuffer(bytes16(0x7BFF), dtype=np.uint8)),
                                            c.__setitem__((slice(None), slice(0, 16)), 0xFF))),
            ("d=subnorm", bytes16(0x0001), lambda c: c.__setitem__((slice(None), slice(0, 16)), 0xFF)),
            ("d=-max", bytes16(0xFBFF), lambda c: c.__setitem__((slice(None), slice(16, 80)), 0xFF)),
            ("distinct-scales", bytes16(0x3C00), lambda c: (c.__setitem__((slice(None), slice(0, 16)),
                                                            np.arange(16, dtype=np.uint8) * np.uint8(0x11)),
                                                            c.__setitem__((slice(None), slice(16, 80)),
                                                                          np.arange(64, dtype=np.uint8)))),
            ("chunk-boundary", bytes16(0x3C00), lambda c: (c.__setitem__((slice(None), slice(16, 48)), 0xA5),
                                                            c.__setitem__((slice(None), slice(48, 80)), 0x5A),
                                                            c.__setitem__((slice(None), slice(0, 16)), 0x0F))),
        ]
    # iq2_xxs
    return [
        ("all-zero-qs", None, lambda c: None),
        ("all-0xFF-qs", None, lambda c: c.__setitem__((slice(None), slice(2, 66)), 0xFF)),
        ("hi-ls=0", bytes16(0x3C00), lambda c: (c.__setitem__((slice(None), slice(2, 66)), 0xFF),
                                                 c.__setitem__((slice(None), slice(6, 10)), 0x00))),
        ("hi-ls=max", bytes16(0x3C00), lambda c: (c.__setitem__((slice(None), slice(2, 66)), 0x00),
                                                   c.__setitem__((slice(None), slice(6, 10)), 0xFF))),
        ("sign-sel-ones", bytes16(0x3C00), lambda c: (c.__setitem__((slice(None), slice(2, 66)), 0x00),
                                                       c.__setitem__((slice(None), slice(3, 6)), 0xFF))),
        ("group7-boundary", bytes16(0x3C00), lambda c: (c.__setitem__((slice(None), slice(58, 66)), 0xFF),
                                                          c.__setitem__((slice(None), slice(60, 64)), 0x00))),
        ("aux8[0]=255", None, lambda c: c.__setitem__((slice(None), slice(2, 4)), 0xFF)),
        ("aux8[last]=255", None, lambda c: c.__setitem__((slice(None), slice(62, 64)), 0xFF)),
        ("d=+max", bytes16(0x7BFF), lambda c: c.__setitem__((slice(None), slice(2, 66)), 0xFF)),
        ("d=subnorm", bytes16(0x0001), lambda c: c.__setitem__((slice(None), slice(2, 66)), 0xFF)),
        ("d=-max", bytes16(0xFBFF), lambda c: c.__setitem__((slice(None), slice(2, 66)), 0xFF)),
    ]


def nonfinite_probe(name):
    if name != "q2_K":
        return None
    c = np.zeros((2, SZ[name]), dtype=np.uint8)
    c[:, 0:16] = 0x0F
    c[0, 80:82] = np.frombuffer(bytes16(0x7C00), dtype=np.uint8)  # +inf
    c[1, 80:82] = np.frombuffer(bytes16(0x7E00), dtype=np.uint8)  # NaN
    return c


# ---------------- harness ----------------------------------------------------

def build_ref_a():
    so = HERE / "ref_a.so"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", f"-I{STUDY}",
                    str(HERE / "ref_a.c"), "-o", str(so)], check=True)
    return so


def load_ref(so):
    lib = ctypes.CDLL(str(so))
    for nm in ("ref_a_q8_0", "ref_a_q2_K", "ref_a_iq2_xxs"):
        getattr(lib, nm).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.ref_a_table.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.ref_a_table.restype = ctypes.c_void_p
    lib.ref_a_sizes.restype = ctypes.POINTER(ctypes.c_size_t)
    lib.ref_a_sizes.argtypes = []
    lib.ref_a_off_qs.restype = ctypes.POINTER(ctypes.c_size_t)
    lib.ref_a_off_qs.argtypes = []
    return lib


def c_dequant(lib, name, raw):
    n, k = raw.shape[0], QK[name]
    y = np.empty(n * k, dtype=F32)
    getattr(lib, f"ref_a_{name}")(raw.ctypes.data, y.ctypes.data, ctypes.c_int64(n * k))
    return y


def tables_from(lib):
    out, hashes = {}, {}
    for name, dtype, nelem in (("iq2xxs_grid", np.uint64, 256),
                               ("ksigns_iq2xs", np.uint8, 128),
                               ("kmask_iq2xs", np.uint8, 8)):
        nbytes = ctypes.c_size_t()
        p = lib.ref_a_table(name.encode(), ctypes.byref(nbytes))
        assert p, name
        buf = (ctypes.c_char * nbytes.value).from_address(p)
        arr = np.frombuffer(bytes(buf), dtype=dtype).copy()
        assert arr.size == nelem, (name, arr.size)
        out[name] = arr
        hashes[name] = hashlib.sha256(arr.tobytes()).hexdigest()
    out["iq2xxs_grid"] = out["iq2xxs_grid"].view(np.uint8).reshape(256, 8)
    return out, hashes


def bitwise_eq(a, b):
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


def nanaware_eq(a, b):
    both = np.isnan(a) & np.isnan(b)
    scrub = lambda x: np.where(both, F32(1.0), x)
    return bitwise_eq(scrub(a), scrub(b))


def run():
    rng = np.random.default_rng(SEED)
    EVID.mkdir(exist_ok=True)
    lib = load_ref(build_ref_a())

    sizes = np.ctypeslib.as_array(lib.ref_a_sizes(), shape=(3,)).tolist()
    off_qs = np.ctypeslib.as_array(lib.ref_a_off_qs(), shape=(3,)).tolist()
    assert sizes == [SZ["q8_0"], SZ["q2_K"], SZ["iq2_xxs"]], sizes
    assert off_qs == [2, 16, 2], off_qs

    tabs, table_hashes = tables_from(lib)
    report = {
        "seed": SEED,
        "n_random_blocks": N_RANDOM_BLOCKS,
        "pinned_source": "Whamp/llama.cpp@0379cf4bf889f3d28038a005210c4bc193fc8ba1",
        "struct_sizes": sizes,
        "qs_offsets": off_qs,
        "table_sha256": table_hashes,
        "formats": {},
    }

    ok_all = True
    for name in ("q8_0", "q2_K", "iq2_xxs"):
        raw = np.ascontiguousarray(gen_random(name, rng, N_RANDOM_BLOCKS))
        rand_ok = bitwise_eq(c_dequant(lib, name, raw), DEC[name](raw, tabs))

        blocks, names = [], []
        for cname, dbytes, mut in adversarial(name):
            c = np.zeros((1, SZ[name]), dtype=np.uint8)
            if dbytes:
                c[0, 0:2] = np.frombuffer(dbytes, dtype=np.uint8)
            mut(c)
            blocks.append(c)
            names.append(cname)
        advb = np.ascontiguousarray(np.concatenate(blocks, 0))
        adv_ok = bitwise_eq(c_dequant(lib, name, advb), DEC[name](advb, tabs))

        nf = nonfinite_probe(name)
        nf_ok = True
        if nf is not None:
            nf = np.ascontiguousarray(nf)
            nf_ok = nanaware_eq(c_dequant(lib, name, nf), DEC[name](nf, tabs))

        entry = {"random_blocks": N_RANDOM_BLOCKS, "random_bitwise_pass": bool(rand_ok),
                 "adversarial_cases": names, "adversarial_bitwise_pass": bool(adv_ok),
                 "nonfinite_nanaware_pass": bool(nf_ok)}
        if not rand_ok:
            ref, mine = c_dequant(lib, name, raw), DEC[name](raw, tabs)
            bad = np.nonzero(ref.view(np.uint32) != mine.view(np.uint32))[0][:5]
            entry["first_random_mismatches"] = [int(i) for i in bad]
            entry["sample_ref"] = [float(ref[i]) for i in bad[:3]]
            entry["sample_mine"] = [float(mine[i]) for i in bad[:3]]
        report["formats"][name] = entry
        ok_all &= rand_ok and adv_ok and nf_ok

    report["pass"] = bool(ok_all)
    (EVID / "l0-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(run())
