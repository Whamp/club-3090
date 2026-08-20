#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8="${PYTHONUTF8:-1}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python3 - <<'PY'
from pathlib import Path

from scripts.lib.profiles.compat import load_profiles
from scripts.lib.profiles.compose_registry import COMPOSE_REGISTRY, DEFAULTS

root = Path.cwd()
canonical_slug = "llamacpp/deepseek-flash-multi4-antirez-iq2-fast-prefill"
canonical_compose = (
    "models/deepseek-v4-flash-0731/llama-cpp/compose/"
    "multi4/antirez-iq2-xxs/fast-prefill.yml"
)

deepseek_variants = {
    slug: entry
    for slug, entry in COMPOSE_REGISTRY.items()
    if entry["model"] == "deepseek-v4-flash-0731" and slug.startswith("llamacpp/")
}
assert set(deepseek_variants) == {canonical_slug}, deepseek_variants.keys()
entry = deepseek_variants[canonical_slug]
assert entry["status"] == "caveats", entry["status"]
assert entry["compose_path"] == canonical_compose, entry["compose_path"]
assert DEFAULTS[("deepseek-v4-flash-0731", "llamacpp", "multi4")] == canonical_slug

profiles = load_profiles()
model = profiles.models["deepseek-v4-flash-0731"]
assert model.default_weight_variant == "antirez-iq2-xxs"
assert set(model.weights) == {"antirez-iq2-xxs"}, model.weights.keys()
assert model.valid_tp == (4,), model.valid_tp
assert model.compatible_drafters == (), model.compatible_drafters

compose_files = sorted(
    path.relative_to(root).as_posix()
    for path in (root / "models/deepseek-v4-flash-0731/llama-cpp/compose").rglob("*.yml")
    if "_archive" not in path.parts
)
assert compose_files == [canonical_compose], compose_files

archived_compose_files = sorted(
    path
    for path in (
        root / "models/deepseek-v4-flash-0731/llama-cpp/compose/_archive"
    ).rglob("*.yml")
)
assert len(archived_compose_files) == 3, archived_compose_files
for archived in archived_compose_files:
    archived_text = archived.read_text()
    assert archived_text.startswith("# ARCHIVED 2026-08-16 — DO NOT LAUNCH.")
    assert "#   Status:    🗑️ Deprecated" in archived_text

setup = (root / "scripts/setup.sh").read_text()
assert 'PRIMARY_WEIGHT_KEY="deepseek-v4-flash-0731:antirez-iq2-xxs"' in setup
assert "llamacpp/deepseek-flash-multi4-antirez-iq2-fast-prefill" in setup
for retired in (
    "llamacpp/deepseek-flash-dual-q8",
    "llamacpp/deepseek-flash-dual-iq2",
    "llamacpp/deepseek-flash-multi4-q8",
):
    assert retired not in setup

litellm = (root / "services/litellm/config.yaml").read_text()
assert "http://host.docker.internal:8033/v1" in litellm
assert "openai/deepseek-v4-flash-0731-q8-fast-prefill" in litellm
PY

echo "PASS: DeepSeek V4 llama.cpp has one canonical Antirez fast-prefill profile"
