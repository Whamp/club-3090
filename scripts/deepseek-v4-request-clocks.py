#!/usr/bin/env python3
"""Lock DeepSeek V4 serving GPUs while a llama.cpp slot is processing."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from types import FrameType


def parse_positive_seconds(value: str) -> float:
    """Parse a strictly positive duration in seconds."""
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than zero")
    return seconds


def parse_arguments() -> argparse.Namespace:
    """Parse the request-aware GPU clock controller command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slots-url", required=True, help="llama.cpp /slots endpoint")
    parser.add_argument("--clock-mhz", type=int, default=1995)
    parser.add_argument(
        "--gpu-indices",
        default="",
        help="comma-separated host GPU indices; empty selects every GPU",
    )
    parser.add_argument("--nvidia-smi", default="/usr/bin/nvidia-smi")
    parser.add_argument(
        "--poll-interval-seconds", type=parse_positive_seconds, default=0.05
    )
    parser.add_argument(
        "--idle-reset-seconds", type=parse_positive_seconds, default=5.0
    )
    parser.add_argument(
        "--endpoint-failure-reset-seconds",
        type=parse_positive_seconds,
        default=5.0,
    )
    parser.add_argument(
        "--endpoint-timeout-seconds", type=parse_positive_seconds, default=1.0
    )
    args = parser.parse_args()
    if args.clock_mhz <= 0:
        parser.error("--clock-mhz must be greater than zero")
    return args


@dataclass(frozen=True)
class DeepSeekV4RequestClockOptions:
    """Configuration for one request-aware GPU clock controller process."""

    slots_url: str
    clock_mhz: int
    gpu_indices: str
    nvidia_smi: str
    poll_interval_seconds: float
    idle_reset_seconds: float
    endpoint_failure_reset_seconds: float
    endpoint_timeout_seconds: float


class DeepSeekV4RequestClockController:
    """Apply an SM clock lock only while the configured llama.cpp slot is busy."""

    def __init__(self, options: DeepSeekV4RequestClockOptions) -> None:
        self.slots_url = options.slots_url
        self.clock_mhz = options.clock_mhz
        self.gpu_indices = options.gpu_indices
        self.nvidia_smi = options.nvidia_smi
        self.poll_interval_seconds = options.poll_interval_seconds
        self.idle_reset_seconds = options.idle_reset_seconds
        self.endpoint_failure_reset_seconds = options.endpoint_failure_reset_seconds
        self.endpoint_timeout_seconds = options.endpoint_timeout_seconds
        self.clock_locked = False
        self.keep_running = True
        self.idle_since: float | None = None
        self.endpoint_failure_since: float | None = None

    def request_stop(self, _signum: int, _frame: FrameType | None) -> None:
        """Request signal-safe loop termination; cleanup runs in the main thread."""
        self.keep_running = False

    def nvidia_smi_command(self, arguments: Sequence[str]) -> list[str]:
        """Build one host NVIDIA clock command for the configured GPU set."""
        command = [self.nvidia_smi]
        if self.gpu_indices:
            command.extend(["-i", self.gpu_indices])
        command.extend(arguments)
        return command

    def run_nvidia_smi(self, *arguments: str) -> None:
        """Run one NVIDIA clock transition and fail if the driver rejects it."""
        subprocess.run(
            self.nvidia_smi_command(arguments),
            check=True,
            stdout=subprocess.DEVNULL,
        )

    def read_any_processing_slot(self) -> bool:
        """Return whether the llama.cpp /slots response contains a busy slot."""
        with urllib.request.urlopen(
            self.slots_url, timeout=self.endpoint_timeout_seconds
        ) as response:
            slots = json.load(response)
        if not isinstance(slots, list):
            raise TypeError("DeepSeek V4 request clocks: /slots response is not a list")
        for slot in slots:
            if not isinstance(slot, dict) or not isinstance(
                slot.get("is_processing"), bool
            ):
                raise TypeError(
                    "DeepSeek V4 request clocks: /slots entry lacks boolean is_processing"
                )
        return any(slot["is_processing"] for slot in slots)

    def lock_sm_clocks(self) -> None:
        """Lock SM clocks for an active request if they are not already locked."""
        if self.clock_locked:
            return
        self.run_nvidia_smi("-lgc", str(self.clock_mhz))
        self.clock_locked = True
        print(
            f"DeepSeek V4 request clocks: locked SM clocks at {self.clock_mhz} MHz",
            flush=True,
        )

    def reset_sm_clocks(self, reason: str) -> None:
        """Reset SM clocks after activity ends or endpoint health is lost."""
        if not self.clock_locked:
            return
        self.run_nvidia_smi("-rgc")
        self.clock_locked = False
        print(f"DeepSeek V4 request clocks: reset SM clocks {reason}", flush=True)

    def update_clock_state(self, *, is_processing: bool, now: float) -> None:
        """Advance the busy/idle clock policy from one valid /slots observation."""
        self.endpoint_failure_since = None
        if is_processing:
            self.idle_since = None
            self.lock_sm_clocks()
            return
        if not self.clock_locked:
            self.idle_since = None
            return
        if self.idle_since is None:
            self.idle_since = now
        if now - self.idle_since >= self.idle_reset_seconds:
            self.reset_sm_clocks("after idle")
            self.idle_since = None

    def update_endpoint_failure(
        self, *, now: float, error: OSError | ValueError | TypeError
    ) -> None:
        """Reset a held lock after a bounded continuous /slots failure."""
        if self.endpoint_failure_since is None:
            self.endpoint_failure_since = now
            print(
                f"DeepSeek V4 request clocks: /slots unavailable: {error}", flush=True
            )
        if (
            self.clock_locked
            and now - self.endpoint_failure_since >= self.endpoint_failure_reset_seconds
        ):
            self.reset_sm_clocks("after endpoint failure")
            self.idle_since = None

    def run(self) -> None:
        """Poll the serving endpoint until signaled, always restoring default clocks."""
        self.run_nvidia_smi("-rgc")
        try:
            while self.keep_running:
                now = time.monotonic()
                try:
                    is_processing = self.read_any_processing_slot()
                except (OSError, ValueError, TypeError) as error:
                    self.update_endpoint_failure(now=now, error=error)
                else:
                    self.update_clock_state(is_processing=is_processing, now=now)
                time.sleep(self.poll_interval_seconds)
        finally:
            self.reset_sm_clocks("during shutdown")


def main() -> None:
    """Run the DeepSeek V4 request-aware GPU clock controller."""
    args = parse_arguments()
    options = DeepSeekV4RequestClockOptions(
        slots_url=args.slots_url,
        clock_mhz=args.clock_mhz,
        gpu_indices=args.gpu_indices,
        nvidia_smi=args.nvidia_smi,
        poll_interval_seconds=args.poll_interval_seconds,
        idle_reset_seconds=args.idle_reset_seconds,
        endpoint_failure_reset_seconds=args.endpoint_failure_reset_seconds,
        endpoint_timeout_seconds=args.endpoint_timeout_seconds,
    )
    controller = DeepSeekV4RequestClockController(options)
    signal.signal(signal.SIGTERM, controller.request_stop)
    signal.signal(signal.SIGINT, controller.request_stop)
    controller.run()


if __name__ == "__main__":
    main()
