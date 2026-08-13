from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_upload_verifier import (
    ArtifactUploadVerification,
    RemoteArtifactFile,
    verify_huggingface_artifact_upload,
)


@dataclass(frozen=True)
class FrontierPublishedCandidate:
    """Immutable Hub result and local cryptographic verification for a candidate."""

    candidate: str
    branch: str
    revision: str
    verification: ArtifactUploadVerification


def ensure_frontier_candidate_branch(
    api: Any,
    repository: str,
    *,
    branch: str,
    parent_revision: str,
    token: str | None,
) -> str:
    """Create a candidate branch or return its existing exact head."""
    api.create_branch(
        repository,
        branch=branch,
        revision=parent_revision,
        token=token,
        exist_ok=True,
    )
    branch_info = api.model_info(repository, revision=branch)
    if not isinstance(branch_info.sha, str):
        raise ValueError(f"frontier branch has no immutable head: {branch}")
    return branch_info.sha


def publish_frontier_candidate(
    directory: Path,
    repository: str,
    *,
    candidate: str,
    branch: str,
    parent_revision: str,
    token: str | None,
    api: Any | None = None,
) -> FrontierPublishedCandidate:
    """Preupload, atomically commit, and verify one immutable candidate."""
    huggingface_hub = _import_optional("huggingface_hub")
    api = api or huggingface_hub.HfApi(token=token)
    parent_info = api.model_info(repository, revision=parent_revision)
    if parent_info.sha != parent_revision:
        raise ValueError(
            f"frontier parent resolved to {parent_info.sha}, expected {parent_revision}"
        )
    branch_head = ensure_frontier_candidate_branch(
        api,
        repository,
        branch=branch,
        parent_revision=parent_revision,
        token=token,
    )
    if branch_head != parent_revision:
        require_frontier_candidate_commit(
            api,
            repository,
            branch=branch,
            candidate=candidate,
            revision=branch_head,
            parent_revision=parent_revision,
            token=token,
        )
        verification = verify_remote_frontier_candidate(
            directory,
            repository,
            revision=branch_head,
            api=api,
        )
        return FrontierPublishedCandidate(
            candidate=candidate,
            branch=branch,
            revision=branch_head,
            verification=verification,
        )

    local_files = _candidate_local_files(directory)
    huggingface_api = _import_optional("huggingface_hub.hf_api")
    inherited_files = {
        item.path
        for item in api.list_repo_tree(
            repository,
            revision=branch,
            recursive=True,
            expand=False,
        )
        if isinstance(item, huggingface_api.RepoFile)
    }
    operations = [
        huggingface_hub.CommitOperationDelete(path_in_repo=relative_path)
        for relative_path in inherited_frontier_paths_to_delete(
            inherited_files,
            set(local_files),
        )
    ]
    for relative_path, path in sorted(local_files.items()):
        addition = huggingface_hub.CommitOperationAdd(
            path_in_repo=relative_path,
            path_or_fileobj=path,
        )
        api.preupload_lfs_files(
            repository,
            additions=[addition],
            revision=branch,
            token=token,
            num_threads=1,
            free_memory=True,
        )
        operations.append(addition)
    commit = api.create_commit(
        repository,
        operations=operations,
        commit_message=f"Add DeepSeek V4 WNA16 frontier candidate {candidate}",
        revision=branch,
        parent_commit=parent_revision,
        token=token,
        num_threads=1,
    )
    verification = verify_remote_frontier_candidate(
        directory,
        repository,
        revision=commit.oid,
        api=api,
    )
    return FrontierPublishedCandidate(
        candidate=candidate,
        branch=branch,
        revision=commit.oid,
        verification=verification,
    )


def require_frontier_candidate_commit(
    api: Any,
    repository: str,
    *,
    branch: str,
    candidate: str,
    revision: str,
    parent_revision: str,
    token: str | None,
) -> None:
    """Require an existing branch to be one exact direct-parent publication."""
    commits = api.list_repo_commits(
        repository,
        revision=branch,
        token=token,
    )
    expected_title = f"Add DeepSeek V4 WNA16 frontier candidate {candidate}"
    if (
        len(commits) < 2
        or commits[0].commit_id != revision
        or commits[0].title != expected_title
        or commits[1].commit_id != parent_revision
    ):
        raise ValueError(f"frontier branch collision or unexpected history: {branch}")


def verify_remote_frontier_candidate(
    local_directory: Path,
    repository: str,
    *,
    revision: str,
    api: Any,
) -> ArtifactUploadVerification:
    """Verify one immutable candidate revision against its local artifact."""
    huggingface_api = _import_optional("huggingface_hub.hf_api")
    model_info = api.model_info(repository, revision=revision)
    if model_info.sha != revision:
        raise ValueError(
            f"frontier candidate revision resolved to {model_info.sha}, "
            f"expected {revision}"
        )
    remote_files = []
    for item in api.list_repo_tree(
        repository,
        revision=revision,
        recursive=True,
        expand=True,
    ):
        if not isinstance(item, huggingface_api.RepoFile):
            continue
        lfs = getattr(item, "lfs", None)
        remote_files.append(
            RemoteArtifactFile(
                path=item.path,
                size=item.size,
                sha256=lfs.sha256 if lfs is not None else None,
                git_blob_sha1=None if lfs is not None else item.blob_id,
            )
        )
    return verify_huggingface_artifact_upload(
        local_directory,
        remote_files,
        repository=repository,
        revision=revision,
    )


def frontier_publish_report_payload(
    published: Iterable[FrontierPublishedCandidate],
    *,
    repository: str | None = None,
    parent_revision: str | None = None,
    recipe_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    """Serialize immutable frontier publication results."""
    return {
        "schema_version": 2,
        "repository": repository,
        "parent_revision": parent_revision,
        "recipe_bundle_sha256": recipe_bundle_sha256,
        "candidates": [
            {
                "candidate": item.candidate,
                "branch": item.branch,
                "revision": item.revision,
                "verification": asdict(item.verification),
            }
            for item in published
        ],
    }


def write_frontier_publish_report(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a frontier publication report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


def inherited_frontier_paths_to_delete(
    inherited_paths: set[str],
    local_paths: set[str],
) -> tuple[str, ...]:
    """Return inherited baseline paths absent from the exact candidate."""
    return tuple(sorted(inherited_paths - local_paths - {".gitattributes"}))


def _candidate_local_files(directory: Path) -> dict[str, Path]:
    files = {}
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if any(part in {".conversion-state", ".cache"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"frontier candidate contains a symlink: {relative}")
        if not path.is_file():
            continue
        files[relative.as_posix()] = path
    expected_shards = {
        shard_name
        for shard_name in json.loads(
            (directory / "model.safetensors.index.json").read_text(encoding="utf-8")
        )["weight_map"].values()
    }
    missing_shards = sorted(expected_shards - set(files))
    if missing_shards:
        raise ValueError(
            f"frontier candidate has missing local shards: {missing_shards[:3]}"
        )
    return files


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for frontier artifact publication"
        ) from error
