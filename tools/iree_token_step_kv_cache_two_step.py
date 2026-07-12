#!/usr/bin/env python3
"""Run the IREE lrrt token-step KV cache VMFB twice and hand off cache output."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Tensor:
    shape: str
    values: list[float]

    def as_input(self) -> str:
        return f"--input={self.shape}=" + " ".join(
            f"{value:g}" for value in self.values
        )


TENSOR_RE = re.compile(r"(?P<shape>\d+x(?:\d+x)?f32)=(?P<body>(?:\[[^\n]*?\])+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_tensors(output: str) -> list[Tensor]:
    tensors: list[Tensor] = []
    for match in TENSOR_RE.finditer(output):
        values = [float(value) for value in NUMBER_RE.findall(match.group("body"))]
        tensors.append(Tensor(match.group("shape"), values))
    if len(tensors) != 3:
        raise ValueError(f"expected 3 output tensors, got {len(tensors)}:\n{output}")
    return tensors


def run_step(
    runner: str,
    module: str,
    query: str,
    key_cache: str,
    new_key: str,
    value_cache: str,
    new_value: str,
) -> list[Tensor]:
    command = [
        runner,
        "--device=lrrt",
        f"--module={module}",
        "--function=token_step_kv_cache_outputs",
        query,
        key_cache,
        new_key,
        value_cache,
        new_value,
        "--input=2xf32=1 0",
        "--input=2xf32=0 1",
        "--output=-",
        "--output=-",
        "--output=-",
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return parse_tensors(completed.stdout)


def expect_close(actual: Tensor, shape: str, expected: list[float], name: str) -> None:
    if actual.shape != shape:
        raise AssertionError(f"{name}: expected shape {shape}, got {actual.shape}")
    if len(actual.values) != len(expected):
        raise AssertionError(
            f"{name}: expected {len(expected)} values, got {len(actual.values)}"
        )
    for index, (observed, wanted) in enumerate(zip(actual.values, expected)):
        if not math.isclose(observed, wanted, rel_tol=1e-5, abs_tol=1e-5):
            raise AssertionError(
                f"{name}[{index}]: expected {wanted:g}, got {observed:g}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--module", required=True)
    args = parser.parse_args()

    first = run_step(
        args.runner,
        args.module,
        "--input=2x2xf32=1 2 3 4",
        "--input=2x3xf32=0 0 0 0 0 0",
        "--input=2xf32=0 0",
        "--input=3x2xf32=2 4 4 6 0 0",
        "--input=2xf32=6 8",
    )
    expect_close(first[0], "2x3xf32", [0, 0, 0, 0, 0, 0], "first key cache")
    expect_close(first[1], "3x2xf32", [2, 4, 4, 6, 6, 8], "first value cache")
    expect_close(first[2], "2x2xf32", [4, 6, 4, 6], "first context")

    second = run_step(
        args.runner,
        args.module,
        "--input=2x2xf32=1 2 3 4",
        first[0].as_input(),
        "--input=2xf32=0 0",
        first[1].as_input(),
        "--input=2xf32=8 10",
    )
    expect_close(second[0], "2x3xf32", [0, 0, 0, 0, 0, 0], "second key cache")
    expect_close(second[1], "3x2xf32", [2, 4, 4, 6, 8, 10], "second value cache")
    expect_close(
        second[2],
        "2x2xf32",
        [14.0 / 3.0, 20.0 / 3.0, 14.0 / 3.0, 20.0 / 3.0],
        "second context",
    )

    print("iree_token_step_kv_cache_two_step: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
