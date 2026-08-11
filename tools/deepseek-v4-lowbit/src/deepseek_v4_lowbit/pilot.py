from __future__ import annotations

import importlib
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from deepseek_v4_lowbit.packing import pack_quantized_tensor
from deepseek_v4_lowbit.quantizer import quantize_symmetric
from deepseek_v4_lowbit.source_dequant import dequantize_routed_expert_weight


@dataclass(frozen=True, order=True)
class PilotSample:
    layer: int
    expert: int
    projection: str

    @property
    def tensor_name(self) -> str:
        return f"layers.{self.layer}.ffn.experts.{self.expert}.{self.projection}.weight"


@dataclass(frozen=True)
class PilotOptions:
    bits: tuple[int, ...] = (2,)
    group_size: int = 128
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not self.bits or any(bit_width not in {2, 4, 8} for bit_width in self.bits):
            raise ValueError("pilot bits must be a non-empty subset of 2, 4, and 8")
        if self.group_size <= 0:
            raise ValueError("pilot group size must be positive")
        if not self.device:
            raise ValueError("pilot device must be non-empty")


@dataclass(frozen=True)
class PilotCandidateResult:
    tensor_name: str
    source_shard: str
    bits: int
    quantizer: str
    duration_seconds: float
    unweighted_error: float
    weighted_error: float


def compare_quantizers(
    source_directory: Path,
    weight_map: dict[str, str],
    imatrix: Any,
    samples: Iterable[PilotSample],
    options: PilotOptions,
) -> tuple[PilotCandidateResult, ...]:
    torch = _import_optional("torch")
    safe_open = _import_optional("safetensors").safe_open
    samples_by_shard: dict[str, list[PilotSample]] = {}
    for sample in sorted(set(samples)):
        if sample.tensor_name not in weight_map:
            raise ValueError(
                f"pilot tensor is absent from source index: {sample.tensor_name}"
            )
        shard_name = weight_map[sample.tensor_name]
        samples_by_shard.setdefault(shard_name, []).append(sample)

    bit_widths = tuple(sorted(set(options.bits)))

    results: list[PilotCandidateResult] = []
    for shard_name, shard_samples in sorted(samples_by_shard.items()):
        shard_path = source_directory / shard_name
        if not shard_path.is_file():
            raise ValueError(f"pilot source shard is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            source_names = set(source.keys())
            for sample in shard_samples:
                scale_name = sample.tensor_name.removesuffix(".weight") + ".scale"
                if scale_name not in source_names:
                    raise ValueError(
                        f"pilot tensor has no source scale: {sample.tensor_name}"
                    )
                dequantized = dequantize_routed_expert_weight(
                    sample.tensor_name,
                    source.get_tensor(sample.tensor_name),
                    source.get_tensor(scale_name),
                    device=options.device,
                ).to(options.device)
                importance = torch.tensor(
                    imatrix.expert_vector(
                        sample.tensor_name,
                        expert_count=256,
                        input_columns=dequantized.shape[1],
                    ),
                    dtype=torch.float32,
                    device=dequantized.device,
                )
                for bit_width in bit_widths:
                    for quantizer_name, optimize_scales in (
                        ("plain-rtn", False),
                        ("imatrix-weighted-rtn", True),
                    ):
                        _synchronize(torch, options.device)
                        started = time.perf_counter()
                        candidate = quantize_symmetric(
                            dequantized,
                            bits=bit_width,
                            group_size=options.group_size,
                            importance=importance,
                            optimize_scales=optimize_scales,
                        )
                        pack_quantized_tensor(
                            candidate,
                            bits=bit_width,
                            group_size=options.group_size,
                        )
                        _synchronize(torch, options.device)
                        duration = time.perf_counter() - started
                        results.append(
                            PilotCandidateResult(
                                tensor_name=sample.tensor_name,
                                source_shard=shard_name,
                                bits=bit_width,
                                quantizer=quantizer_name,
                                duration_seconds=duration,
                                unweighted_error=candidate.unweighted_error,
                                weighted_error=candidate.weighted_error,
                            )
                        )
    return tuple(results)


def pilot_results_payload(
    results: Iterable[PilotCandidateResult],
) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"{module_name} is required for the quantizer pilot"
        ) from error
