from __future__ import annotations

import importlib
import json
import os
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.artifact_upload_verifier import (
    ArtifactUploadVerification,
)
from deepseek_v4_lowbit.frontier_convert import (
    FrontierCandidateConversion,
    convert_frontier_candidates,
    load_completed_frontier_candidate,
    validate_frontier_candidate_names,
)
from deepseek_v4_lowbit.frontier_manifest import (
    build_frontier_candidate_manifest,
    write_frontier_candidate_manifest,
    write_frontier_candidate_model_card,
)
from deepseek_v4_lowbit.frontier_publish import (
    FrontierPublishedCandidate,
    publish_frontier_candidate,
    require_frontier_candidate_commit,
    verify_remote_frontier_candidate,
)
from deepseek_v4_lowbit.shard_writer import file_sha256


class FrontierBatchPublisher:
    """Publish one completed candidate before optionally reclaiming local disk."""

    def __init__(
        self,
        repository: str,
        *,
        parent_revision: str,
        branch_prefix: str,
        recipe_bundle_path: Path,
        report_path: Path,
        token: str,
        delete_local_after_verify: bool,
        candidate_summaries: dict[str, dict[str, Any]],
        candidate_names: tuple[str, ...],
        evidence_files: tuple[Path, ...] = (),
    ) -> None:
        self.repository = repository
        self.parent_revision = parent_revision
        self.branch_prefix = branch_prefix
        self.recipe_bundle_path = recipe_bundle_path
        self.recipe_bundle_sha256 = file_sha256(recipe_bundle_path)
        self.report_path = report_path
        self.token = token
        self.delete_local_after_verify = delete_local_after_verify
        self.candidate_summaries = candidate_summaries
        self.candidate_names = validate_frontier_candidate_names(candidate_names)
        if len(self.candidate_names) != 1:
            raise ValueError("frontier campaign must select exactly one candidate")
        self.evidence_files = evidence_files
        self.published = self._load_published_prefix()
        self._previous_directory: Path | None = None

    def __call__(self, conversion: FrontierCandidateConversion) -> None:
        """Manifest, publish, verify, record, then optionally delete a candidate."""
        summary = self.candidate_summaries[conversion.name]
        recipe_destination = conversion.output_directory / "frontier-recipe-bundle.json"
        temporary_recipe = recipe_destination.with_name(
            f".{recipe_destination.name}.writing"
        )
        shutil.copyfile(self.recipe_bundle_path, temporary_recipe)
        os.replace(temporary_recipe, recipe_destination)
        if file_sha256(recipe_destination) != self.recipe_bundle_sha256:
            raise ValueError("frontier candidate recipe bundle copy checksum mismatch")
        evidence_directory = conversion.output_directory / "frontier-evidence"
        evidence_directory.mkdir(exist_ok=True)
        for source_path in self.evidence_files:
            destination = evidence_directory / source_path.name
            temporary = destination.with_name(f".{destination.name}.writing")
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, destination)
            if file_sha256(destination) != file_sha256(source_path):
                raise ValueError(
                    f"frontier candidate evidence copy checksum mismatch: {source_path}"
                )
        write_frontier_candidate_model_card(
            conversion.output_directory / "README.md",
            candidate=conversion.name,
            summary=summary,
            parent_revision=self.parent_revision,
            recipe_bundle_sha256=self.recipe_bundle_sha256,
        )
        manifest_path = conversion.output_directory / "frontier-manifest.json"
        write_frontier_candidate_manifest(
            manifest_path,
            build_frontier_candidate_manifest(
                conversion.output_directory,
                candidate=conversion.name,
                parent_revision=self.parent_revision,
                recipe_bundle_sha256=self.recipe_bundle_sha256,
                recipe_sha256=conversion.recipe_sha256,
            ),
        )
        published = publish_frontier_candidate(
            conversion.output_directory,
            self.repository,
            candidate=conversion.name,
            branch=f"{self.branch_prefix}/{conversion.name}",
            parent_revision=self.parent_revision,
            token=self.token,
        )
        self.published.append(published)
        self._write_report()
        if (
            self.delete_local_after_verify
            and self._previous_directory is not None
            and self._previous_directory != conversion.output_directory
        ):
            shutil.rmtree(self._previous_directory)
        self._previous_directory = conversion.output_directory

    def finalize_local_storage(self) -> None:
        """Delete the final verified candidate after all nested reuse is complete."""
        if self.delete_local_after_verify and self._previous_directory is not None:
            shutil.rmtree(self._previous_directory)
            self._previous_directory = None

    def _write_report(self) -> None:
        payload = {
            "schema_version": 3,
            "repository": self.repository,
            "parent_revision": self.parent_revision,
            "recipe_bundle_sha256": self.recipe_bundle_sha256,
            "selected_candidate_names": list(self.candidate_names),
            "candidates": [
                {
                    "candidate": item.candidate,
                    "branch": item.branch,
                    "revision": item.revision,
                    "verification": {
                        field.name: getattr(item.verification, field.name)
                        for field in fields(item.verification)
                    },
                }
                for item in self.published
            ],
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_name(f".{self.report_path.name}.writing")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as file_handle:
            os.fsync(file_handle.fileno())
        os.replace(temporary, self.report_path)

    def resume_conversion_state(
        self,
        source_directory: Path,
        output_root: Path,
        recipe_bundle: dict[str, Any],
        imatrix_path: Path,
        *,
        device: str,
    ) -> tuple[tuple[str, ...], FrontierCandidateConversion | None]:
        """Verify the published prefix and recover its latest local reuse base."""
        completed_names = tuple(item.candidate for item in self.published)
        if not self.published:
            return completed_names, None
        latest = self.published[-1]
        latest_directory = output_root / latest.candidate
        if (
            len(self.published) == len(self.candidate_names)
            and not latest_directory.exists()
        ):
            return completed_names, None
        local_candidate = load_completed_frontier_candidate(
            source_directory,
            output_root,
            recipe_bundle,
            candidate_name=latest.candidate,
            imatrix_path=imatrix_path,
            device=device,
        )
        verification = verify_remote_frontier_candidate(
            local_candidate.output_directory,
            self.repository,
            revision=latest.revision,
            api=self._hub_api(),
        )
        if verification != latest.verification:
            raise ValueError("frontier resume verification differs from publish report")
        if self.delete_local_after_verify:
            for published in self.published[:-1]:
                stale_directory = output_root / published.candidate
                if stale_directory.exists():
                    shutil.rmtree(stale_directory)
        self._previous_directory = local_candidate.output_directory
        return completed_names, local_candidate

    def _load_published_prefix(self) -> list[FrontierPublishedCandidate]:
        if not self.report_path.exists():
            return []
        payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 3
            or payload.get("repository") != self.repository
            or payload.get("parent_revision") != self.parent_revision
            or payload.get("recipe_bundle_sha256") != self.recipe_bundle_sha256
            or payload.get("selected_candidate_names") != list(self.candidate_names)
        ):
            raise ValueError("frontier publication report provenance mismatch")
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list) or len(raw_candidates) > len(
            self.candidate_names
        ):
            raise ValueError("frontier publication report candidates are invalid")
        published = []
        for index, raw in enumerate(raw_candidates):
            if (
                not isinstance(raw, dict)
                or raw.get("candidate") != self.candidate_names[index]
            ):
                raise ValueError("frontier publication report is not an ordered prefix")
            verification = raw.get("verification")
            if not isinstance(verification, dict):
                raise ValueError("frontier publication report verification is invalid")
            published_candidate = FrontierPublishedCandidate(
                candidate=raw["candidate"],
                branch=raw["branch"],
                revision=raw["revision"],
                verification=self._verification_from_payload(verification),
            )
            require_frontier_candidate_commit(
                self._hub_api(),
                self.repository,
                branch=published_candidate.branch,
                candidate=published_candidate.candidate,
                revision=published_candidate.revision,
                parent_revision=self.parent_revision,
                token=self.token,
            )
            published.append(published_candidate)
        return published

    def _hub_api(self) -> Any:
        huggingface_hub = importlib.import_module("huggingface_hub")
        return huggingface_hub.HfApi(token=self.token)

    @staticmethod
    def _verification_from_payload(
        payload: dict[str, Any],
    ) -> ArtifactUploadVerification:
        normalized = dict(payload)
        managed_files = normalized.get("hub_managed_files")
        if isinstance(managed_files, list):
            normalized["hub_managed_files"] = tuple(managed_files)
        try:
            return ArtifactUploadVerification(**normalized)
        except TypeError as error:
            raise ValueError(
                "frontier publication report verification fields are invalid"
            ) from error


def run_frontier_conversion_batch(
    source_directory: Path,
    output_root: Path,
    recipe_bundle_path: Path,
    imatrix_path: Path,
    baseline_directory: Path,
    *,
    device: str,
    publisher: FrontierBatchPublisher,
    gpu_devices: tuple[str, ...] = (),
) -> tuple[FrontierCandidateConversion, ...]:
    """Convert and durably publish the campaign's one selected candidate."""
    recipe_bundle = json.loads(recipe_bundle_path.read_text(encoding="utf-8"))
    if not isinstance(recipe_bundle, dict):
        raise ValueError("frontier recipe bundle must be a JSON object")
    completed_names, reuse_candidate = publisher.resume_conversion_state(
        source_directory,
        output_root,
        recipe_bundle,
        imatrix_path,
        device=device,
    )
    if len(completed_names) == len(publisher.candidate_names):
        publisher.finalize_local_storage()
        return ()
    converted = convert_frontier_candidates(
        source_directory,
        output_root,
        recipe_bundle,
        baseline_directory=baseline_directory,
        imatrix_path=imatrix_path,
        device=device,
        completed_callback=publisher,
        candidate_names=publisher.candidate_names,
        completed_candidate_names=completed_names,
        reuse_candidate=reuse_candidate,
        gpu_devices=gpu_devices,
    )
    publisher.finalize_local_storage()
    return converted
