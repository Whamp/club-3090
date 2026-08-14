from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_lowbit.artifact_upload_verifier import (
    RemoteArtifactFile,
    git_blob_sha1,
    verify_huggingface_artifact_upload,
)


class ArtifactUploadVerifierTests(unittest.TestCase):
    def test_verifies_lfs_and_git_blob_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory)
            shard = artifact / "model-00001.safetensors"
            config = artifact / "config.json"
            state = artifact / ".conversion-state" / "receipt.json"
            shard.write_bytes(b"safetensors payload")
            config.write_text('{"model_type":"deepseek_v4"}\n')
            state.parent.mkdir()
            state.write_text("not uploaded")
            shard_sha256 = hashlib.sha256(shard.read_bytes()).hexdigest()

            verification = verify_huggingface_artifact_upload(
                artifact,
                [
                    RemoteArtifactFile(
                        shard.name,
                        shard.stat().st_size,
                        shard_sha256,
                        None,
                    ),
                    RemoteArtifactFile(
                        config.name,
                        config.stat().st_size,
                        None,
                        git_blob_sha1(config),
                    ),
                    RemoteArtifactFile(".gitattributes", 1, None, "ignored"),
                ],
                repository="hampsonw/test-artifact",
                revision="abc123",
            )

            self.assertEqual(verification.file_count, 2)
            self.assertEqual(verification.sha256_file_count, 1)
            self.assertEqual(verification.git_blob_file_count, 1)
            self.assertEqual(verification.hub_managed_files, (".gitattributes",))

    def test_ignores_local_hub_managed_gitattributes_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory)
            config = artifact / "config.json"
            attributes = artifact / ".gitattributes"
            config.write_text("{}\n")
            attributes.write_text("local attributes\n")

            verification = verify_huggingface_artifact_upload(
                artifact,
                [
                    RemoteArtifactFile(
                        config.name,
                        config.stat().st_size,
                        None,
                        git_blob_sha1(config),
                    ),
                    RemoteArtifactFile(
                        attributes.name,
                        999,
                        None,
                        "hub-managed-content",
                    ),
                ],
                repository="hampsonw/test-artifact",
                revision="abc123",
            )

            self.assertEqual(verification.file_count, 1)
            self.assertEqual(verification.total_bytes, config.stat().st_size)
            self.assertEqual(verification.hub_managed_files, (".gitattributes",))

    def test_rejects_inventory_size_and_hash_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact = Path(temporary_directory)
            config = artifact / "config.json"
            config.write_text("{}\n")

            with self.assertRaisesRegex(ValueError, "inventory mismatch"):
                verify_huggingface_artifact_upload(
                    artifact,
                    [],
                    repository="hampsonw/test-artifact",
                    revision="abc123",
                )
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_huggingface_artifact_upload(
                    artifact,
                    [RemoteArtifactFile(config.name, 999, None, "bad")],
                    repository="hampsonw/test-artifact",
                    revision="abc123",
                )
            with self.assertRaisesRegex(ValueError, "Git blob mismatch"):
                verify_huggingface_artifact_upload(
                    artifact,
                    [
                        RemoteArtifactFile(
                            config.name,
                            config.stat().st_size,
                            None,
                            "bad",
                        )
                    ],
                    repository="hampsonw/test-artifact",
                    revision="abc123",
                )


if __name__ == "__main__":
    unittest.main()
