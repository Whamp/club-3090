#!/usr/bin/env bash
set -euo pipefail

export PYTHONUTF8="${PYTHONUTF8:-1}"
unset CUDA_VISIBLE_DEVICES

readonly CLUB_3090_REVISION="4213eeae4b1fc45b68014e729d2dbd74ff950b6c"
readonly CLUB_3090_REF="refs/heads/feat/deepseek-v4-quant-frontier"
readonly AUTO_ROUND_REVISION="f17d9cd4b36982006bad21ff87127aac739072e3"
readonly DEEPSEEK_REVISION="7872f01b1d1fe23eabc4c98b48bffcef5a386062"
readonly BASELINE_REVISION="75d9286c37f3037f3ab390cfbc10747466eac714"
readonly IMATRIX_REVISION="e7f04037032990db0346398d249baf9fb9df1ccc"
readonly IMATRIX_SHA256="02a7c78c29875e4653d6ce21d8821c02161e83ed90c506bdd8d275f76d4ac97e"
readonly SOURCE_REPOSITORY="deepseek-ai/DeepSeek-V4-Flash-0731"
readonly BASELINE_REPOSITORY="hampsonw/DeepSeek-V4-Flash-0731-WNA16"
readonly IMATRIX_REPOSITORY="antirez/deepseek-v4-gguf"
readonly IMATRIX_FILENAME="imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat"
readonly UV_VERSION="0.9.28"
readonly TORCH_VERSION="2.13.0"
readonly COMPRESSED_TENSORS_VERSION="0.17.0"
readonly HUGGINGFACE_REPOSITORY="${2:-$BASELINE_REPOSITORY}"
readonly BRANCH_PREFIX="${3:-frontier-20260813}"
readonly FRONTIER_CANDIDATE="${4:-quality}"
readonly RENTAL_ROOT="${1:-$HOME/deepseek-v4-quant-frontier}"
export HF_HOME="$RENTAL_ROOT/huggingface-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
readonly SOURCE_DIRECTORY="$RENTAL_ROOT/source/DeepSeek-V4-Flash-0731"
readonly BASELINE_DIRECTORY="$RENTAL_ROOT/baseline"
readonly IMATRIX_DIRECTORY="$RENTAL_ROOT/imatrix"
readonly IMATRIX_PATH="$IMATRIX_DIRECTORY/$IMATRIX_FILENAME"
readonly REPORT_DIRECTORY="$RENTAL_ROOT/reports"
readonly SCREEN_DIRECTORY="$REPORT_DIRECTORY/frontier-screen"
readonly SOURCE_HEADERS_REPORT="$REPORT_DIRECTORY/source-headers-report.json"
readonly PLANNER_HEADERS="$REPORT_DIRECTORY/source-headers.json"
readonly RECIPE_BUNDLE="$REPORT_DIRECTORY/frontier-recipe-bundle.json"
readonly PUBLICATION_REPORT="$REPORT_DIRECTORY/frontier-publication.json"
readonly COMPLETION_RECEIPT="$REPORT_DIRECTORY/frontier-complete.json"
readonly OUTPUT_ROOT="$RENTAL_ROOT/output"
readonly CLUB_3090_DIRECTORY="$RENTAL_ROOT/club-3090"
readonly AUTO_ROUND_DIRECTORY="$RENTAL_ROOT/auto-round"
readonly PYTHON_ENVIRONMENT="$RENTAL_ROOT/python-environment"
readonly RENTAL_LOG="$REPORT_DIRECTORY/run-verda-quant-frontier.log"
readonly -a FRONTIER_GPU_DEVICES=(0)
gpu_arguments=()
for device in "${FRONTIER_GPU_DEVICES[@]}"; do
    gpu_arguments+=(--gpu-device "$device")
done
readonly -a gpu_arguments

mkdir -p \
    "$REPORT_DIRECTORY" \
    "$SOURCE_DIRECTORY" \
    "$BASELINE_DIRECTORY" \
    "$HF_HUB_CACHE" \
    "$HF_XET_CACHE"
exec > >(tee -a "$RENTAL_LOG") 2>&1

log_frontier_step() {
    printf '\n[%s] %s\n' "$(date --utc +%Y-%m-%dT%H:%M:%SZ)" "$1"
}

require_clean_checkout() {
    local destination="$1"
    [[ -z "$(git -C "$destination" status --porcelain --untracked-files=all)" ]] || {
        echo "Pinned frontier checkout must be clean: $destination" >&2
        return 2
    }
}

checkout_pinned_repository() {
    local repository_url="$1"
    local repository_ref="$2"
    local revision="$3"
    local destination="$4"
    if [[ -d "$destination/.git" ]]; then
        require_clean_checkout "$destination"
    else
        git clone --filter=blob:none --no-checkout "$repository_url" "$destination"
    fi
    git -C "$destination" fetch --depth 1 origin "$repository_ref"
    git -C "$destination" checkout --detach "$revision"
    test "$(git -C "$destination" rev-parse HEAD)" = "$revision"
    require_clean_checkout "$destination"
}

[[ "$CLUB_3090_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
    echo "Frontier rental runner requires a full published Git revision" >&2
    exit 2
}
[[ -n "${HF_TOKEN:-}" ]] || {
    echo "Frontier rental runner requires HF_TOKEN" >&2
    exit 2
}
case "$FRONTIER_CANDIDATE" in
    cliff | capacity | balanced | quality) ;;
    *)
        echo "Unknown frontier candidate: $FRONTIER_CANDIDATE" >&2
        exit 2
        ;;
esac

log_frontier_step "Verify bounded rental hardware, memory, disk, and Hub access"
python3 - "$RENTAL_ROOT" <<'PY'
import shutil
import sys
free_gib = shutil.disk_usage(sys.argv[1]).free / (1024**3)
if free_gib < 280:
    raise SystemExit(
        f"Frontier rental requires at least 280 GiB free, found {free_gib:.1f}"
    )
print(f"free_disk_gib={free_gib:.1f}")
PY
gpu_inventory="$(nvidia-smi \
    --query-gpu=name,compute_cap,memory.total \
    --format=csv,noheader,nounits)"
python3 - "$gpu_inventory" <<'PY'
import sys
rows = [row.strip() for row in sys.argv[1].splitlines() if row.strip()]
if len(rows) != 1:
    raise SystemExit(
        "Frontier rental requires exactly one A100 80GB GPU, "
        f"found {len(rows)}"
    )
for row in rows:
    name, capability, memory_mib = (field.strip() for field in row.split(",", 2))
    if "A100" not in name or capability != "8.0" or int(memory_mib) < 80_000:
        raise SystemExit(
            "Frontier rental requires exactly one A100 80GB GPU at "
            f"compute capability 8.0, found {row}"
        )
print("gpu_contract=1x-a100-80gb-sm80")
PY
memory_kib="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
((memory_kib >= 110 * 1024 * 1024)) || {
    echo "Frontier rental requires at least 110 GiB host RAM" >&2
    exit 2
}

log_frontier_step "Install pinned uv and conversion environment"
if ! command -v uv >/dev/null || [[ "$(uv --version)" != "uv $UV_VERSION" ]]; then
    curl --fail --location --silent --show-error \
        "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
checkout_pinned_repository \
    "https://github.com/Whamp/club-3090.git" \
    "$CLUB_3090_REF" \
    "$CLUB_3090_REVISION" \
    "$CLUB_3090_DIRECTORY"
checkout_pinned_repository \
    "https://github.com/intel/auto-round.git" \
    "refs/heads/main" \
    "$AUTO_ROUND_REVISION" \
    "$AUTO_ROUND_DIRECTORY"
uv venv --allow-existing --python 3.12 "$PYTHON_ENVIRONMENT"
uv pip install --python "$PYTHON_ENVIRONMENT/bin/python" \
    "torch==$TORCH_VERSION" \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PYTHON_ENVIRONMENT/bin/python" \
    --no-build-isolation \
    --editable "$AUTO_ROUND_DIRECTORY"
uv pip install --python "$PYTHON_ENVIRONMENT/bin/python" \
    "compressed-tensors==$COMPRESSED_TENSORS_VERSION" \
    "huggingface_hub[cli]==1.27.0" \
    --editable "$CLUB_3090_DIRECTORY/tools/deepseek-v4-lowbit"
uv pip check --python "$PYTHON_ENVIRONMENT/bin/python"

log_frontier_step "Verify four dedicated spawned CUDA workers"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-inspect-frontier-gpus" \
    "${gpu_arguments[@]}"

log_frontier_step "Verify exact selected CUDA device and write-capable public Hub target"
"$PYTHON_ENVIRONMENT/bin/python" - "$HUGGINGFACE_REPOSITORY" <<'PY'
import os
import sys
import torch
from huggingface_hub import HfApi
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Frontier rental requires one visible CUDA device")
for device_index in range(torch.cuda.device_count()):
    if torch.cuda.get_device_capability(device_index) != (8, 0):
        raise SystemExit("Frontier rental requires compute capability 8.0")
    if "A100" not in torch.cuda.get_device_name(device_index):
        raise SystemExit("Frontier rental selected CUDA devices must be A100s")
print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"gpus={torch.cuda.device_count()} "
    f"gpu0={torch.cuda.get_device_name(0)} "
    f"cc0={torch.cuda.get_device_capability(0)}"
)
repository_id = sys.argv[1]
namespace = repository_id.partition("/")[0]
api = HfApi(token=os.environ["HF_TOKEN"])
identity = api.whoami()
access_token = (identity.get("auth") or {}).get("accessToken") or {}
role = access_token.get("role")
has_write_access = role == "write"
if role == "fineGrained":
    for scope in (access_token.get("fineGrained") or {}).get("scoped", []):
        entity = scope.get("entity") or {}
        permissions = set(scope.get("permissions") or [])
        if entity.get("name") == namespace and "repo.write" in permissions:
            has_write_access = True
            break
if not has_write_access:
    raise SystemExit(f"HF_TOKEN lacks repo.write for namespace {namespace!r}")
repository = api.model_info(repository_id)
if repository.private:
    raise SystemExit(f"Frontier target must remain public: {repository_id}")
print(
    f"huggingface_user={identity.get('name')} repository={repository.id} "
    "private=false write_access=true"
)
PY

log_frontier_step "Download pinned official checkpoint and routed-expert imatrix"
"$PYTHON_ENVIRONMENT/bin/hf" download \
    "$SOURCE_REPOSITORY" \
    --revision "$DEEPSEEK_REVISION" \
    --local-dir "$SOURCE_DIRECTORY" \
    --max-workers 8
mkdir -p "$IMATRIX_DIRECTORY"
"$PYTHON_ENVIRONMENT/bin/hf" download \
    "$IMATRIX_REPOSITORY" \
    "$IMATRIX_FILENAME" \
    --revision "$IMATRIX_REVISION" \
    --local-dir "$IMATRIX_DIRECTORY" \
    --max-workers 1
printf '%s  %s\n' "$IMATRIX_SHA256" "$IMATRIX_PATH" | \
    sha256sum --check --strict
hf_cache_bytes="$(du --summarize --bytes "$HF_HOME" | cut -f1)"
((hf_cache_bytes <= 2 * 1024 * 1024 * 1024)) || {
    echo "Hugging Face cache unexpectedly consumed $hf_cache_bytes bytes" >&2
    echo "Frontier downloads must not duplicate model shards in cache" >&2
    exit 2
}

log_frontier_step "Capture exact source headers, shards, and copied asset hashes"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-capture-source-headers" \
    "$SOURCE_DIRECTORY" \
    "$SOURCE_HEADERS_REPORT" \
    "$PLANNER_HEADERS"

log_frontier_step "Download immutable baseline metadata"
"$PYTHON_ENVIRONMENT/bin/hf" download \
    "$BASELINE_REPOSITORY" \
    config.json \
    model.safetensors.index.json \
    conversion-metrics.json \
    --revision "$BASELINE_REVISION" \
    --local-dir "$BASELINE_DIRECTORY" \
    --max-workers 1

log_frontier_step "Run and stabilize mixed-group frontier screen"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-run-frontier-screen" \
    "$SOURCE_DIRECTORY" \
    "$IMATRIX_PATH" \
    "$BASELINE_DIRECTORY/conversion-metrics.json" \
    "$SOURCE_HEADERS_REPORT" \
    "$PLANNER_HEADERS" \
    "$SCREEN_DIRECTORY" \
    --samples-per-projection 8 \
    --device cuda \
    "${gpu_arguments[@]}"

log_frontier_step "Build exact cliff/capacity/balanced/quality recipes"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-build-frontier-recipes" \
    "$BASELINE_DIRECTORY/conversion-metrics.json" \
    "$SCREEN_DIRECTORY/frontier-pilot-screen.json" \
    "$SCREEN_DIRECTORY/frontier-boundary-report.json" \
    "$SCREEN_DIRECTORY/frontier-full-screen.json" \
    "$SOURCE_HEADERS_REPORT" \
    "$PLANNER_HEADERS" \
    "$RECIPE_BUNDLE" \
    --model-parameter-count 284334567511

log_frontier_step "Verify measured source plus projected transition disk bound"
"$PYTHON_ENVIRONMENT/bin/python" - \
    "$SOURCE_DIRECTORY" "$BASELINE_DIRECTORY" "$RECIPE_BUNDLE" "$RENTAL_ROOT" \
    "$FRONTIER_CANDIDATE" <<'PY'
import json
import shutil
import sys
from pathlib import Path
source, baseline, recipe_path, rental_root = map(Path, sys.argv[1:5])
selected_candidate = sys.argv[5]
recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
source_bytes = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
baseline_bytes = sum(
    path.stat().st_size for path in baseline.rglob("*") if path.is_file()
)
matching_summaries = [
    summary
    for summary in recipe["candidate_summaries"]
    if summary.get("name") == selected_candidate
]
if len(matching_summaries) != 1:
    raise SystemExit("Frontier recipe has no unique selected-candidate summary")
candidate_payload = int(matching_summaries[0]["total_bytes"])
required = source_bytes + baseline_bytes + candidate_payload + 8 * 1024**3
free = shutil.disk_usage(rental_root).free
if free + source_bytes + baseline_bytes < required:
    raise SystemExit(
        "Frontier projected disk bound exceeds rental capacity: "
        f"candidate={selected_candidate} required_gib={required / 1024**3:.2f} "
        f"capacity_gib={(free + source_bytes + baseline_bytes) / 1024**3:.2f}"
    )
print(
    f"source_gib={source_bytes / 1024**3:.2f} "
    f"candidate={selected_candidate} "
    f"selected_candidate_gib={candidate_payload / 1024**3:.2f} "
    f"required_with_margin_gib={required / 1024**3:.2f}"
)
PY

log_frontier_step "Download only baseline shards selected for exact reuse"
mapfile -t reusable_shards < <(
    "$PYTHON_ENVIRONMENT/bin/python" - "$RECIPE_BUNDLE" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for shard in payload["storage_summary"]["baseline_reused_shard_names"]:
    print(shard)
PY
)
((${#reusable_shards[@]} > 0)) || {
    echo "Frontier recipe selected no reusable baseline shards" >&2
    exit 2
}
"$PYTHON_ENVIRONMENT/bin/hf" download \
    "$BASELINE_REPOSITORY" \
    "${reusable_shards[@]}" \
    --revision "$BASELINE_REVISION" \
    --local-dir "$BASELINE_DIRECTORY" \
    --max-workers 4

log_frontier_step "Convert, publish, verify, and reclaim selected candidate"
"$PYTHON_ENVIRONMENT/bin/deepseek-v4-run-frontier-batch" \
    "$SOURCE_DIRECTORY" \
    "$OUTPUT_ROOT" \
    "$RECIPE_BUNDLE" \
    "$IMATRIX_PATH" \
    "$BASELINE_DIRECTORY" \
    "$HUGGINGFACE_REPOSITORY" \
    "$PUBLICATION_REPORT" \
    --parent-revision "$BASELINE_REVISION" \
    --branch-prefix "$BRANCH_PREFIX" \
    --candidate "$FRONTIER_CANDIDATE" \
    --device cuda \
    "${gpu_arguments[@]}" \
    --delete-local-after-verify \
    --evidence-file "$SOURCE_HEADERS_REPORT" \
    --evidence-file "$PLANNER_HEADERS" \
    --evidence-file "$SCREEN_DIRECTORY/frontier-pilot-screen.json" \
    --evidence-file "$SCREEN_DIRECTORY/frontier-screen-iterations.json" \
    --evidence-file "$SCREEN_DIRECTORY/frontier-boundary-report.json" \
    --evidence-file "$SCREEN_DIRECTORY/frontier-full-screen.json"

log_frontier_step "Verify the immutable published revision after local deletion"
"$PYTHON_ENVIRONMENT/bin/python" - \
    "$HUGGINGFACE_REPOSITORY" "$PUBLICATION_REPORT" "$BASELINE_REVISION" \
    "$FRONTIER_CANDIDATE" "$COMPLETION_RECEIPT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repository, report_path, parent, expected_candidate, receipt_path = sys.argv[1:]
report_bytes = Path(report_path).read_bytes()
report = json.loads(report_bytes)
api = HfApi(token=os.environ["HF_TOKEN"])
if (
    report["parent_revision"] != parent
    or report.get("selected_candidate_names") != [expected_candidate]
    or len(report["candidates"]) != 1
    or report["candidates"][0].get("candidate") != expected_candidate
):
    raise SystemExit("Frontier publication report is incomplete or misidentified")
candidate = report["candidates"][0]
revision = candidate["revision"]
if api.model_info(repository, revision=revision).sha != revision:
    raise SystemExit(f"Frontier revision no longer resolves: {revision}")
commits = api.list_repo_commits(repository, revision=candidate["branch"])
if commits[0].commit_id != revision or commits[1].commit_id != parent:
    raise SystemExit(f"Frontier branch history drift: {candidate['branch']}")
receipt = {
    "schema_version": 1,
    "candidate": expected_candidate,
    "revision": revision,
    "branch": candidate["branch"],
    "parent_revision": parent,
    "publication_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
}
receipt_path = Path(receipt_path)
temporary = receipt_path.with_name(f".{receipt_path.name}.writing")
temporary.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
os.replace(temporary, receipt_path)
directory_fd = os.open(receipt_path.parent, os.O_RDONLY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(
    f"verified candidate={candidate['candidate']} revision={revision} "
    f"bytes={candidate['verification']['total_bytes']}"
)
PY

log_frontier_step "Frontier rental campaign complete"
sha256sum \
    "$SOURCE_HEADERS_REPORT" \
    "$PLANNER_HEADERS" \
    "$SCREEN_DIRECTORY/frontier-pilot-screen.json" \
    "$SCREEN_DIRECTORY/frontier-boundary-report.json" \
    "$SCREEN_DIRECTORY/frontier-full-screen.json" \
    "$RECIPE_BUNDLE" \
    "$PUBLICATION_REPORT" \
    "$COMPLETION_RECEIPT"
printf 'publication_report=%s\nrecipe_bundle=%s\nrental_log=%s\n' \
    "$PUBLICATION_REPORT" "$RECIPE_BUNDLE" "$RENTAL_LOG"
