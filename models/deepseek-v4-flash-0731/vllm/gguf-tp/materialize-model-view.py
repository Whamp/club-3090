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

For an Unsloth IQ1 variant, quantization_config also includes the exact
620-entry source-type profile and its checksum. Existing Antirez behavior is
unchanged when no variant is supplied.

Every output config.json sha256 is pinned — regeneration must reproduce it
byte-for-byte or the script fails.

Usage:
  materialize-model-view.py <artifact_snapshot_dir> <blobs_dir> <out_dir> [UD-IQ1_S|UD-IQ1_M]
"""
import hashlib
import json
import pathlib
import sys

ARTIFACT_CONFIG = "config.json"
EXPECTED_BASE_CONFIG_SHA256 = (
    "334bfa9f35a2f05510639538325c20e87a3980b06cabef4d750d3ca8085a0a66"
)
EXPECTED_CONFIG_SHA256 = "e973e27b89f47929848a7360bd05b624fd2141f0af2451d9d67636e359c2b4cb"
SOURCE_PROFILES = pathlib.Path(__file__).with_name("UNSLOTH-IQ1-SOURCE-PROFILES.json")
EXPECTED_SOURCE_PROFILES_SHA256 = (
    "367c489edb3390b75f4d290bd2bcf85cccb48841c0bcbe04fc8c0bf4ea6d6c75"
)
EXPECTED_VARIANT_CONFIG_SHA256 = {
    "UD-IQ1_S": "4693c91fd050bc8a566abef30f3b900fc439a0233ae1b576ddc7793064695511",
    "UD-IQ1_M": "506cf2197bf240be6914ea50039b3664f72a8b14bcd5eec15465b44f9e4ae4cf",
}

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
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    snapshot, blobs, out = (pathlib.Path(a) for a in sys.argv[1:4])
    variant = sys.argv[4] if len(sys.argv) == 5 else None
    if variant is not None and variant not in EXPECTED_VARIANT_CONFIG_SHA256:
        raise SystemExit(f"FAIL: unsupported GGUF variant {variant!r}")

    cfg_src = snapshot / ARTIFACT_CONFIG
    if not cfg_src.exists():
        raise SystemExit(f"FAIL: artifact config not found at {cfg_src}")
    require_sha256(cfg_src, EXPECTED_BASE_CONFIG_SHA256, "base config.json")
    src = json.loads(cfg_src.read_text())

    derived = dict(src)
    quantization_config = {"quant_method": "gguf_dsv4"}
    if variant is not None:
        require_sha256(
            SOURCE_PROFILES,
            EXPECTED_SOURCE_PROFILES_SHA256,
            "Unsloth IQ1 source profiles",
        )
        profiles = json.loads(SOURCE_PROFILES.read_text())
        profile = profiles[variant]
        quantization_config.update(profile)
    derived["quantization_config"] = quantization_config
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
    cfg_out.write_text(json.dumps(derived, indent=2, ensure_ascii=False))
    expected_config_sha256 = (
        EXPECTED_CONFIG_SHA256
        if variant is None
        else EXPECTED_VARIANT_CONFIG_SHA256[variant]
    )
    require_sha256(cfg_out, expected_config_sha256, "derived config.json")
    label = "Antirez" if variant is None else variant
    print(
        f"OK: {label} model view materialized at {out} "
        f"(config.json {expected_config_sha256[:12]}…)"
    )


if __name__ == "__main__":
    main()
