from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.shard_writer import file_sha256

_HUB_MANAGED_FILES = {".gitattributes"}
_LOCAL_STATE_DIRECTORIES = {".conversion-state", ".cache"}


@dataclass(frozen=True)
class RemoteArtifactFile:
    """Metadata required to verify one uploaded Hugging Face artifact file."""

    path: str
    size: int
    sha256: str | None
    git_blob_sha1: str | None


@dataclass(frozen=True)
class ArtifactUploadVerification:
    """Cryptographic verification summary for a Hugging Face artifact upload."""

    repository: str
    revision: str
    file_count: int
    total_bytes: int
    sha256_file_count: int
    git_blob_file_count: int
    hub_managed_files: tuple[str, ...]


def verify_huggingface_artifact_upload(
    local_directory: Path,
    remote_files: Iterable[RemoteArtifactFile],
    *,
    repository: str,
    revision: str,
) -> ArtifactUploadVerification:
    """Verify exact artifact inventory, sizes, and remote content hashes."""
    local_files = _artifact_local_files(local_directory)
    remote_file_list = tuple(remote_files)
    remote_by_path = {remote.path: remote for remote in remote_file_list}
    if len(remote_by_path) != len(remote_file_list):
        raise ValueError(
            "Hugging Face upload verification found duplicate remote paths"
        )

    local_names = set(local_files)
    remote_names = set(remote_by_path)
    missing = sorted(local_names - remote_names)
    extra = sorted(remote_names - local_names - _HUB_MANAGED_FILES)
    if missing or extra:
        raise ValueError(
            f"Hugging Face upload inventory mismatch: missing={missing}, extra={extra}"
        )

    total_bytes = 0
    sha256_file_count = 0
    git_blob_file_count = 0
    for relative_path, local_path in sorted(local_files.items()):
        remote = remote_by_path[relative_path]
        local_size = local_path.stat().st_size
        if remote.size != local_size:
            raise ValueError(
                f"Hugging Face upload size mismatch for {relative_path}: "
                f"local={local_size}, remote={remote.size}"
            )
        if remote.sha256 is not None:
            local_hash = file_sha256(local_path)
            if local_hash != remote.sha256:
                raise ValueError(
                    f"Hugging Face upload SHA-256 mismatch for {relative_path}"
                )
            sha256_file_count += 1
        elif remote.git_blob_sha1 is not None:
            local_hash = git_blob_sha1(local_path)
            if local_hash != remote.git_blob_sha1:
                raise ValueError(
                    f"Hugging Face upload Git blob mismatch for {relative_path}"
                )
            git_blob_file_count += 1
        else:
            raise ValueError(
                f"Hugging Face upload has no content hash for {relative_path}"
            )
        total_bytes += local_size

    return ArtifactUploadVerification(
        repository=repository,
        revision=revision,
        file_count=len(local_files),
        total_bytes=total_bytes,
        sha256_file_count=sha256_file_count,
        git_blob_file_count=git_blob_file_count,
        hub_managed_files=tuple(sorted(remote_names & _HUB_MANAGED_FILES)),
    )


def git_blob_sha1(path: Path) -> str:
    """Return the Git blob object ID for one ordinary uploaded file."""
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as file_handle:
        while block := file_handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Cryptographically verify a Hugging Face artifact upload."
    )
    parser.add_argument("local_directory", type=Path)
    parser.add_argument("repository")
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args(argv)

    huggingface_hub = _import_optional("huggingface_hub")
    huggingface_api = _import_optional("huggingface_hub.hf_api")
    api = huggingface_hub.HfApi()
    model_info = api.model_info(arguments.repository)
    revision = model_info.sha
    remote_files = []
    for item in api.list_repo_tree(
        arguments.repository,
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

    verification = verify_huggingface_artifact_upload(
        arguments.local_directory.resolve(),
        remote_files,
        repository=arguments.repository,
        revision=revision,
    )
    _write_json_atomic(arguments.report.resolve(), asdict(verification))
    print(
        f"verified upload: repository={verification.repository} "
        f"revision={verification.revision} files={verification.file_count} "
        f"bytes={verification.total_bytes}"
    )
    return 0


def _artifact_local_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"artifact upload directory does not exist: {directory}")
    files: dict[str, Path] = {}
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if any(part in _LOCAL_STATE_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"artifact upload contains a symlink: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path
    if not files:
        raise ValueError(f"artifact upload directory contains no files: {directory}")
    return files


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    os.replace(temporary, path)


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required to verify the artifact upload"
        ) from error


if __name__ == "__main__":
    raise SystemExit(main())
