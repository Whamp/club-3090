from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_lowbit.imatrix import ImatrixFile, map_hf_expert_to_imatrix


class ImatrixNameMappingTests(unittest.TestCase):
    def test_maps_official_projection_names_to_ds4_entries(self) -> None:
        self.assertEqual(
            map_hf_expert_to_imatrix("layers.26.ffn.experts.17.w1.weight"),
            ("blk.26.ffn_gate_exps.weight", 17),
        )
        self.assertEqual(
            map_hf_expert_to_imatrix("layers.26.ffn.experts.17.w3.weight"),
            ("blk.26.ffn_up_exps.weight", 17),
        )
        self.assertEqual(
            map_hf_expert_to_imatrix("layers.26.ffn.experts.17.w2.weight"),
            ("blk.26.ffn_down_exps.weight", 17),
        )

    def test_rejects_non_routed_tensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a routed expert weight"):
            map_hf_expert_to_imatrix("layers.26.ffn.shared_experts.w2.weight")


class ImatrixFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "imatrix.dat"
        _write_imatrix(
            self.path,
            {
                "blk.0.ffn_gate_exps.weight": (2, [float(value) for value in range(8)]),
                "blk.0.ffn_up_exps.weight": (1, [1.0] * 8),
                "blk.0.ffn_down_exps.weight": (1, [2.0] * 4),
            },
            chunks=7,
            dataset="rendered-prompts.txt",
        )

    def test_indexes_without_copying_and_slices_one_expert(self) -> None:
        with ImatrixFile.open(self.path) as imatrix:
            vector = imatrix.expert_vector(
                "layers.0.ffn.experts.1.w1.weight",
                expert_count=2,
                input_columns=4,
            )

            self.assertEqual(vector, (2.0, 2.5, 3.0, 3.5))
            self.assertEqual(imatrix.entry_count, 3)
            self.assertEqual(imatrix.chunks, 7)
            self.assertEqual(imatrix.dataset, "rendered-prompts.txt")

    def test_rejects_wrong_expert_geometry(self) -> None:
        with (
            ImatrixFile.open(self.path) as imatrix,
            self.assertRaisesRegex(ValueError, "value count mismatch"),
        ):
            imatrix.expert_vector(
                "layers.0.ffn.experts.1.w1.weight",
                expert_count=2,
                input_columns=8,
            )

    def test_rejects_duplicate_entry_names(self) -> None:
        duplicate = Path(self.temp_dir.name) / "duplicate.dat"
        _write_entries(
            duplicate,
            [
                ("blk.0.ffn_gate_exps.weight", 1, [1.0]),
                ("blk.0.ffn_gate_exps.weight", 1, [2.0]),
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate imatrix entry"):
            ImatrixFile.open(duplicate)

    def test_rejects_truncated_values(self) -> None:
        self.path.write_bytes(self.path.read_bytes()[:-12])

        with self.assertRaisesRegex(ValueError, "truncated imatrix"):
            ImatrixFile.open(self.path)


def _write_imatrix(
    path: Path,
    entries: dict[str, tuple[int, list[float]]],
    *,
    chunks: int,
    dataset: str,
) -> None:
    _write_entries(
        path,
        [(name, calls, values) for name, (calls, values) in entries.items()],
        trailer=(chunks, dataset),
    )


def _write_entries(
    path: Path,
    entries: list[tuple[str, int, list[float]]],
    trailer: tuple[int, str] | None = None,
) -> None:
    with path.open("wb") as output:
        output.write(struct.pack("<i", len(entries)))
        for name, calls, values in entries:
            encoded_name = name.encode("utf-8")
            output.write(struct.pack("<i", len(encoded_name)))
            output.write(encoded_name)
            output.write(struct.pack("<ii", calls, len(values)))
            output.write(struct.pack(f"<{len(values)}f", *values))
        if trailer is not None:
            chunks, dataset = trailer
            encoded_dataset = dataset.encode("utf-8")
            output.write(struct.pack("<ii", chunks, len(encoded_dataset)))
            output.write(encoded_dataset)


if __name__ == "__main__":
    unittest.main()
