from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_lowbit.source_headers import capture_source_tensor_headers

_HAS_DEPS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None
)


@unittest.skipUnless(_HAS_DEPS, "requires torch and safetensors")
class SourceTensorHeadersTests(unittest.TestCase):
    def test_captures_indexed_raw_headers(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            shard_name = "model-00001-of-00001.safetensors"
            shard_path = root / shard_name
            save_file({"model.weight": torch.ones((2, 3))}, shard_path)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.weight": shard_name}}),
                encoding="utf-8",
            )
            (root / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"}),
                encoding="utf-8",
            )

            headers = capture_source_tensor_headers(root)

        record = headers[shard_name]["model.weight"]
        self.assertEqual(record["dtype"], "F32")
        self.assertEqual(record["shape"], [2, 3])
        self.assertEqual(record["data_offsets"], [0, 24])


if __name__ == "__main__":
    unittest.main()
