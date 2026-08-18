#!/usr/bin/env python3
"""Fail-closed materializer for the GGUF-TP engine model view.

The engine loads a *derived* model view (plain directory) plus the raw GGUF
blob: config.json tells vLLM the architecture/quantization, the other small
files come from the WNA16 artifact's immutable blobs via symlinks (git-blob
SHA-1 names, so they are content-addressed), and weights come from the GGUF
file itself (--load-format gguf_dsv4).

Derivation from the uniform WNA16 artifact config (rev 75d9286c):
  1. quantization_config -> {"quant_method": "gguf_dsv4"} (wholesale replace)
  2. top-level quant_method -> "gguf_dsv4" (new key)
  3. club_3090_lowbit.source_quantization_method "fp8" -> "compressed-tensors"

Output config.json sha256 is pinned (e973e27b…) — regeneration must reproduce
it byte-for-byte or the script fails.

Usage:
  materialize-model-view.py <artifact_snapshot_dir> <blobs_dir> <out_dir>
"""
import hashlib
import json
import pathlib
import sys

ARTIFACT_CONFIG = "config.json"
EXPECTED_CONFIG_SHA256 = "e973e27b89f47929848a7360bd05b624fd2141f0af2451d9d67636e359c2b4cb"

# git-blob SHA-1 names inside the HF cache blobs/ dir (content-addressed).
BLOB_SYMLINKS = {
    "LICENSE": "d62e3bef9f054f21b7fc616365850fbf879a99ff",
    "README.md": "d11eeca20136244e643dcb26d4d3a31b979a9ba6",
    "generation_config.json": "c56a8c5bf06ff7740b6a33ee67b38b6237a230b1",
    "tokenizer.json": "628e3364caad11bdf9e67cea06eae7878122811d",
    "tokenizer_config.json": "f3dad388a2bbfd6a8605bd02754acd86d9ca5112",
}


def require_sha256(path: pathlib.Path, expected: str, label: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"FAIL {label}: sha256 {actual} != {expected}")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    snapshot, blobs, out = (pathlib.Path(a) for a in sys.argv[1:])

    cfg_src = snapshot / ARTIFACT_CONFIG
    if not cfg_src.exists():
        raise SystemExit(f"FAIL: artifact config not found at {cfg_src}")
    src = json.loads(cfg_src.read_text())

    derived = dict(src)
    derived["quantization_config"] = {"quant_method": "gguf_dsv4"}
    derived["quant_method"] = "gguf_dsv4"
    lb = derived.get("club_3090_lowbit")
    if lb and lb.get("source_quantization_method") == "fp8":
        lb["source_quantization_method"] = "compressed-tensors"

    out.mkdir(parents=True, exist_ok=True)
    for name, blob in BLOB_SYMLINKS.items():
        target = blobs / blob
        if not target.exists():
            raise SystemExit(f"FAIL: blob {blob} (for {name}) not in {blobs}")
        link = out / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)

    cfg_out = out / ARTIFACT_CONFIG
    cfg_out.write_text(json.dumps(derived, indent=4, ensure_ascii=False) + "\n")
    require_sha256(cfg_out, EXPECTED_CONFIG_SHA256, "derived config.json")
    print(f"OK: model view materialized at {out} (config.json {EXPECTED_CONFIG_SHA256[:12]}…)")


if __name__ == "__main__":
    main()
