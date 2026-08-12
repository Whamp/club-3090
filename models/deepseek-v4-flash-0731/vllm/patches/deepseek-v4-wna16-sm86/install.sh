#!/usr/bin/env bash
set -euo pipefail

readonly PINNED_BASE="12810046c799cbe874967e19b1c0fa134ab7b209"
readonly EXPECTED_FINAL_TREE="9a54d487051e937a9dd6c146b971d93ff422eb30"
PATCH_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PATCH_DIRECTORY
readonly TARGET_REPOSITORY="${1:?usage: install.sh /path/to/vllm}"

[[ -d "$TARGET_REPOSITORY/.git" || -f "$TARGET_REPOSITORY/.git" ]] || {
    echo "DeepSeek V4 WNA16 patch target is not a Git worktree: $TARGET_REPOSITORY" >&2
    exit 2
}
[[ -z "$(git -C "$TARGET_REPOSITORY" status --porcelain)" ]] || {
    echo "DeepSeek V4 WNA16 patch target must be clean: $TARGET_REPOSITORY" >&2
    exit 2
}

current_tree="$(git -C "$TARGET_REPOSITORY" rev-parse 'HEAD^{tree}')"
if [[ "$current_tree" == "$EXPECTED_FINAL_TREE" ]]; then
    echo "DeepSeek V4 WNA16 patch series already applied"
    exit 0
fi

current_revision="$(git -C "$TARGET_REPOSITORY" rev-parse HEAD)"
[[ "$current_revision" == "$PINNED_BASE" ]] || {
    echo "DeepSeek V4 WNA16 patch base mismatch: got $current_revision" >&2
    echo "Expected pinned haosdent base: $PINNED_BASE" >&2
    exit 2
}

if ! git -C "$TARGET_REPOSITORY" am \
    --committer-date-is-author-date \
    "$PATCH_DIRECTORY"/*.patch; then
    git -C "$TARGET_REPOSITORY" am --abort
    echo "DeepSeek V4 WNA16 patch series failed; git am was aborted" >&2
    exit 1
fi

actual_final_tree="$(git -C "$TARGET_REPOSITORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_final_tree" == "$EXPECTED_FINAL_TREE" ]] || {
    echo "DeepSeek V4 WNA16 patched tree mismatch: got $actual_final_tree" >&2
    echo "Expected tree: $EXPECTED_FINAL_TREE" >&2
    exit 1
}

echo "DeepSeek V4 WNA16 patch series applied: tree=$actual_final_tree"
