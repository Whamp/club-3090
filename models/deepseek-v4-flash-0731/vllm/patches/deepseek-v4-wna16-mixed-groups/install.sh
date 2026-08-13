#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_PARENT_TREE="aeb62948e33074514a742d19c2f9a1a3c2ee3e1f"
readonly EXPECTED_FINAL_TREE="f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538"
SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIRECTORY
readonly PATCH_PATH="$SCRIPT_DIRECTORY/0009-feat-support-mixed-WNA16-MoE-projection-groups.patch"
readonly TARGET_REPOSITORY="${1:?usage: install.sh /path/to/eight-patch-vllm}"

[[ -d "$TARGET_REPOSITORY/.git" || -f "$TARGET_REPOSITORY/.git" ]] || {
    echo "DeepSeek V4 mixed-group target is not a Git worktree: $TARGET_REPOSITORY" >&2
    exit 2
}
[[ -z "$(git -C "$TARGET_REPOSITORY" status --porcelain --untracked-files=all)" ]] || {
    echo "DeepSeek V4 mixed-group target must be clean: $TARGET_REPOSITORY" >&2
    exit 2
}

current_tree="$(git -C "$TARGET_REPOSITORY" rev-parse 'HEAD^{tree}')"
if [[ "$current_tree" == "$EXPECTED_FINAL_TREE" ]]; then
    echo "DeepSeek V4 mixed-group extension already applied"
    exit 0
fi
[[ "$current_tree" == "$EXPECTED_PARENT_TREE" ]] || {
    echo "DeepSeek V4 mixed-group parent tree mismatch: got $current_tree" >&2
    echo "Expected accepted eight-patch tree: $EXPECTED_PARENT_TREE" >&2
    exit 2
}

if ! git -C "$TARGET_REPOSITORY" am \
    --committer-date-is-author-date \
    "$PATCH_PATH"; then
    git -C "$TARGET_REPOSITORY" am --abort
    echo "DeepSeek V4 mixed-group extension failed; git am was aborted" >&2
    exit 1
fi

actual_final_tree="$(git -C "$TARGET_REPOSITORY" rev-parse 'HEAD^{tree}')"
[[ "$actual_final_tree" == "$EXPECTED_FINAL_TREE" ]] || {
    echo "DeepSeek V4 mixed-group tree mismatch: got $actual_final_tree" >&2
    echo "Expected tree: $EXPECTED_FINAL_TREE" >&2
    exit 1
}

echo "DeepSeek V4 mixed-group extension applied: tree=$actual_final_tree"
