from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from deepseek_v4_lowbit.artifact_plan import ArtifactRecipe, LayerQuantization
from deepseek_v4_lowbit.artifact_upload_verifier import ArtifactUploadVerification
from deepseek_v4_lowbit.frontier_batch import FrontierBatchPublisher
from deepseek_v4_lowbit.frontier_convert import FrontierCandidateConversion
from deepseek_v4_lowbit.frontier_manifest import write_frontier_candidate_model_card
from deepseek_v4_lowbit.frontier_publish import (
    FrontierPublishedCandidate,
    _candidate_local_files,
    ensure_frontier_candidate_branch,
    inherited_frontier_paths_to_delete,
    publish_frontier_candidate,
    require_frontier_candidate_commit,
)
from deepseek_v4_lowbit.frontier_publish_cli import main as publish_frontier_main
from deepseek_v4_lowbit.shard_writer import file_sha256


class FrontierPublishTests(unittest.TestCase):
    def test_candidate_card_pins_runtime_and_pending_behavioral_gate(self) -> None:
        summary = {
            "w13_group128_layers": [],
            "w13_group256_layers": [],
            "w13_group512_layers": [0],
            "w2_group128_layers": [0],
            "w2_group256_layers": [],
            "w2_group512_layers": [],
            "w4_down_layers": [],
            "total_bytes": 100,
            "total_gib": 100 / 1024**3,
            "whole_model_bits_per_parameter": 2.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            card_path = Path(temporary_directory) / "README.md"
            write_frontier_candidate_model_card(
                card_path,
                candidate="quality",
                summary=summary,
                parent_revision="a" * 40,
                recipe_bundle_sha256="b" * 64,
            )
            card = card_path.read_text(encoding="utf-8")

        self.assertIn("dd2d1fd6779addccc73094f77fa4ada7d9106a41", card)
        self.assertIn("f73b30cc5a2ed9de200ca2e4de3cdef1a06f6538", card)
        self.assertIn("has not passed the single-worker DeepSWE gate", card)

    def test_collects_recursive_candidate_files_without_conversion_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "frontier-evidence").mkdir()
            (root / "frontier-evidence" / "screen.json").write_text("{}\n")
            (root / ".conversion-state").mkdir()
            (root / ".conversion-state" / "receipt.json").write_text("{}\n")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"model.weight":"model.safetensors"}}\n'
            )
            (root / "model.safetensors").write_bytes(b"payload")

            files = _candidate_local_files(root)

        self.assertIn("frontier-evidence/screen.json", files)
        self.assertNotIn(".conversion-state/receipt.json", files)

    def test_deletes_inherited_files_absent_from_exact_candidate(self) -> None:
        self.assertEqual(
            inherited_frontier_paths_to_delete(
                {".gitattributes", "old-config.json", "shared.bin"},
                {"shared.bin", "new-config.json"},
            ),
            ("old-config.json",),
        )

    def test_standalone_publisher_requires_one_explicit_candidate(self) -> None:
        verification = _verification("c" * 40)
        published = FrontierPublishedCandidate(
            candidate="quality",
            branch="frontier/quality",
            revision="c" * 40,
            verification=verification,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "quality").mkdir()
            with (
                patch.dict("os.environ", {"HF_TOKEN": "token"}),
                patch(
                    "deepseek_v4_lowbit.frontier_publish_cli.publish_frontier_candidate",
                    return_value=published,
                ) as publish,
                patch(
                    "deepseek_v4_lowbit.frontier_publish_cli.write_frontier_publish_report"
                ) as write_report,
            ):
                status = publish_frontier_main(
                    [
                        str(root),
                        "owner/repository",
                        str(root / "report.json"),
                        "--candidate",
                        "quality",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertEqual(publish.call_args.kwargs["candidate"], "quality")
        self.assertEqual(
            publish.call_args.kwargs["branch"],
            "frontier/quality",
        )
        report_payload = write_report.call_args.args[1]
        self.assertEqual(
            [candidate["candidate"] for candidate in report_payload["candidates"]],
            ["quality"],
        )

    def test_ensures_candidate_branch_from_exact_parent(self) -> None:
        api = Mock()
        api.model_info.return_value = SimpleNamespace(sha="a" * 40)

        head = ensure_frontier_candidate_branch(
            api,
            "owner/repository",
            branch="frontier/cliff",
            parent_revision="a" * 40,
            token="token",
        )

        self.assertEqual(head, "a" * 40)
        api.create_branch.assert_called_once_with(
            "owner/repository",
            branch="frontier/cliff",
            revision="a" * 40,
            token="token",
            exist_ok=True,
        )

    def test_requires_existing_candidate_to_have_exact_parent_history(self) -> None:
        api = Mock()
        api.list_repo_commits.return_value = [
            SimpleNamespace(commit_id="c" * 40, title="unrelated"),
            SimpleNamespace(commit_id="a" * 40, title="parent"),
        ]

        with self.assertRaisesRegex(ValueError, "collision or unexpected history"):
            require_frontier_candidate_commit(
                api,
                "owner/repository",
                branch="frontier/cliff",
                candidate="cliff",
                revision="c" * 40,
                parent_revision="a" * 40,
                token="token",
            )

    def test_resumes_exact_existing_candidate_without_uploading(self) -> None:
        api = Mock()
        api.model_info.side_effect = [
            SimpleNamespace(sha="a" * 40),
            SimpleNamespace(sha="c" * 40),
        ]
        api.list_repo_commits.return_value = [
            SimpleNamespace(
                commit_id="c" * 40,
                title="Add DeepSeek V4 WNA16 frontier candidate cliff",
            ),
            SimpleNamespace(commit_id="a" * 40, title="parent"),
        ]
        verification = _verification("c" * 40)
        with patch(
            "deepseek_v4_lowbit.frontier_publish.verify_remote_frontier_candidate",
            return_value=verification,
        ):
            published = publish_frontier_candidate(
                Path("/unused"),
                "owner/repository",
                candidate="cliff",
                branch="frontier/cliff",
                parent_revision="a" * 40,
                token="token",
                api=api,
            )

        self.assertEqual(published.revision, "c" * 40)
        api.preupload_lfs_files.assert_not_called()
        api.create_commit.assert_not_called()

    def test_batch_publisher_rejects_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recipe_bundle = root / "frontier-recipe-bundle.json"
            recipe_bundle.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one candidate"):
                FrontierBatchPublisher(
                    "owner/repository",
                    parent_revision="a" * 40,
                    branch_prefix="frontier",
                    recipe_bundle_path=recipe_bundle,
                    report_path=root / "report.json",
                    token="token",
                    delete_local_after_verify=True,
                    candidate_summaries={},
                    candidate_names=("cliff", "quality"),
                )

    def test_deletes_prior_candidate_only_after_next_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            previous = root / "cliff"
            current = root / "capacity"
            previous.mkdir()
            current.mkdir()
            publisher = _publisher_without_init(root)
            publisher._previous_directory = previous
            conversion = FrontierCandidateConversion(
                name="capacity",
                output_directory=current,
                recipe=ArtifactRecipe(default=LayerQuantization(2, 2)),
                recipe_sha256="r" * 64,
                results=(),
                reused_shard_count=0,
                converted_shard_count=0,
            )
            with (
                patch(
                    "deepseek_v4_lowbit.frontier_batch.write_frontier_candidate_model_card"
                ),
                patch(
                    "deepseek_v4_lowbit.frontier_batch.build_frontier_candidate_manifest",
                    return_value={},
                ),
                patch(
                    "deepseek_v4_lowbit.frontier_batch.write_frontier_candidate_manifest"
                ),
                patch.object(publisher, "_write_report"),
                patch(
                    "deepseek_v4_lowbit.frontier_batch.publish_frontier_candidate",
                    side_effect=RuntimeError("verification failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "verification failed"),
            ):
                publisher(conversion)
            self.assertTrue(previous.exists())
            self.assertTrue(current.exists())

            published = FrontierPublishedCandidate(
                candidate="capacity",
                branch="frontier/capacity",
                revision="d" * 40,
                verification=_verification("d" * 40),
            )
            with (
                patch(
                    "deepseek_v4_lowbit.frontier_batch.write_frontier_candidate_model_card"
                ),
                patch(
                    "deepseek_v4_lowbit.frontier_batch.build_frontier_candidate_manifest",
                    return_value={},
                ),
                patch(
                    "deepseek_v4_lowbit.frontier_batch.write_frontier_candidate_manifest"
                ),
                patch.object(publisher, "_write_report"),
                patch(
                    "deepseek_v4_lowbit.frontier_batch.publish_frontier_candidate",
                    return_value=published,
                ),
            ):
                publisher(conversion)
            self.assertFalse(previous.exists())
            self.assertTrue(current.exists())

            publisher.finalize_local_storage()
            self.assertFalse(current.exists())


def _publisher_without_init(root: Path) -> FrontierBatchPublisher:
    publisher = FrontierBatchPublisher.__new__(FrontierBatchPublisher)
    publisher.repository = "owner/repository"
    publisher.parent_revision = "a" * 40
    publisher.branch_prefix = "frontier"
    publisher.recipe_bundle_path = root / "frontier-recipe-bundle.json"
    publisher.recipe_bundle_path.write_text("{}\n", encoding="utf-8")
    publisher.recipe_bundle_sha256 = file_sha256(publisher.recipe_bundle_path)
    publisher.report_path = root / "publish.json"
    publisher.token = "token"
    publisher.delete_local_after_verify = True
    publisher.candidate_summaries = {"capacity": {}}
    publisher.candidate_names = ("capacity",)
    publisher.evidence_files = ()
    publisher.published = []
    publisher._previous_directory = None
    return publisher


def _verification(revision: str) -> ArtifactUploadVerification:
    return ArtifactUploadVerification(
        repository="owner/repository",
        revision=revision,
        file_count=2,
        total_bytes=100,
        sha256_file_count=1,
        git_blob_file_count=1,
        hub_managed_files=(".gitattributes",),
    )


if __name__ == "__main__":
    unittest.main()
