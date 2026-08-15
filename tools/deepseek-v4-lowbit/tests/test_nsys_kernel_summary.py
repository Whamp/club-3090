from __future__ import annotations

import pytest

from deepseek_v4_lowbit.nsys_kernel_summary import summarize_nsys_cuda_kernels


def test_nsys_kernel_summary_separates_nccl_from_total(tmp_path) -> None:
    report = tmp_path / "kernels.csv"
    report.write_text(
        "Time (%),Total Time (ns),Instances,Name\n"
        '25.0,250,4,"ncclDevKernel_AllReduce_RING_LL"\n'
        '75.0,750,8,"sparse_mla_decode_kernel"\n',
        encoding="utf-8",
    )

    summary = summarize_nsys_cuda_kernels(report)

    assert summary["scope"] == "summed_cuda_kernel_time_not_critical_path"
    assert summary["nccl_kernel_time_fraction"] == pytest.approx(0.25)
    assert summary["nccl_kernels"] == [
        {"name": "ncclDevKernel_AllReduce_RING_LL", "total_time": 250.0}
    ]


def test_nsys_kernel_summary_rejects_missing_header(tmp_path) -> None:
    report = tmp_path / "invalid.csv"
    report.write_text("not,a,kernel,summary\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no Total Time/Name header"):
        summarize_nsys_cuda_kernels(report)
