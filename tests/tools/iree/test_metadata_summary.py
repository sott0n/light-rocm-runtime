#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "tools" / "iree_metadata_summary.py"


FIXTURE = """
module {
  util.global private @__device_0 = #hal.device.target<"hip", [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1101", features = "">}>]> : !hal.device
  hal.executable private @simple_mul_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {}>) {
      hal.executable.export public @simple_mul_dispatch_0_elementwise_4_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      ^bb0(%arg0: !hal.device):
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      } attributes {subgroup_size = 32 : index, workgroup_size = [32 : index, 1 : index, 1 : index]}
      builtin.module {
        llvm.func @simple_mul_dispatch_0_elementwise_4_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {gpu.known_block_size = array<i32: 32, 1, 1>, rocdl.flat_work_group_size = "32,32", rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 32, 1, 1>} {
          llvm.return
        }
      }
    }
  }
  stream.cmd.dispatch @simple_mul_dispatch_0::@rocm_hsaco_fb::@simple_mul_dispatch_0_elementwise_4_f32 {
  }
}
"""

MULTI_EXECUTABLE_FIXTURE = """
module {
  util.global private @__device_0 = #hal.device.target<"hip", [#hal.executable.target<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #iree_gpu.target<arch = "gfx1101", features = "">}>]> : !hal.device
  hal.executable private @mixed_matmuls_dispatch_0 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {}>) {
      hal.executable.export public @mixed_matmuls_dispatch_0_matmul_2x2x2_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      ^bb0(%arg0: !hal.device):
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      } attributes {subgroup_size = 32 : index, workgroup_size = [2 : index, 2 : index, 1 : index]}
      builtin.module {
        llvm.func @mixed_matmuls_dispatch_0_matmul_2x2x2_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {gpu.known_block_size = array<i32: 2, 2, 1>, rocdl.flat_work_group_size = "4,4", rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 2, 2, 1>} {
          llvm.return
        }
      }
    }
  }
  hal.executable private @mixed_matmuls_dispatch_1 {
    hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {}>) {
      hal.executable.export public @mixed_matmuls_dispatch_1_matmul_2x3x2_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      ^bb0(%arg0: !hal.device):
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      } attributes {subgroup_size = 32 : index, workgroup_size = [2 : index, 3 : index, 1 : index]}
      builtin.module {
        llvm.func @mixed_matmuls_dispatch_1_matmul_2x3x2_f32(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {gpu.known_block_size = array<i32: 2, 3, 1>, rocdl.flat_work_group_size = "6,6", rocdl.kernel, rocdl.reqd_work_group_size = array<i32: 2, 3, 1>} {
          llvm.return
        }
      }
    }
  }
  stream.cmd.dispatch @mixed_matmuls_dispatch_0::@rocm_hsaco_fb::@mixed_matmuls_dispatch_0_matmul_2x2x2_f32 {
  }
  stream.cmd.dispatch @mixed_matmuls_dispatch_1::@rocm_hsaco_fb::@mixed_matmuls_dispatch_1_matmul_2x3x2_f32 {
  }
}
"""


def load_summary():
    spec = importlib.util.spec_from_file_location("iree_metadata_summary", SUMMARY)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_metadata() -> None:
    module = load_summary()
    summary = module.parse_iree_metadata(FIXTURE)

    assert summary["target"] == "gfx1101"
    assert len(summary["executables"]) == 1
    executable = summary["executables"][0]
    assert executable["executable"] == "simple_mul_dispatch_0"
    assert executable["variant"] == "rocm_hsaco_fb"

    export = executable["exports"][0]
    assert export["symbol"] == "simple_mul_dispatch_0_elementwise_4_f32"
    assert export["ordinal"] == 0
    assert export["workgroup_size"] == [32, 1, 1]
    assert export["subgroup_size"] == 32
    assert export["bindings"] == [
        {"index": 0, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
        {"index": 1, "type": "storage_buffer", "flags": ["ReadOnly", "Indirect"]},
        {"index": 2, "type": "storage_buffer", "flags": ["Indirect"]},
    ]
    assert export["kernel"]["symbol"] == export["symbol"]
    assert "rocdl.kernel" in export["kernel"]["attributes"]
    assert export["dispatch"] == {
        "executable": "simple_mul_dispatch_0",
        "variant": "rocm_hsaco_fb",
        "symbol": "simple_mul_dispatch_0_elementwise_4_f32",
    }


def test_parse_multi_executable_metadata() -> None:
    module = load_summary()
    summary = module.parse_iree_metadata(MULTI_EXECUTABLE_FIXTURE)

    assert summary["target"] == "gfx1101"
    assert [item["executable"] for item in summary["executables"]] == [
        "mixed_matmuls_dispatch_0",
        "mixed_matmuls_dispatch_1",
    ]
    assert summary["executables"][0]["exports"][0]["symbol"] == (
        "mixed_matmuls_dispatch_0_matmul_2x2x2_f32"
    )
    assert summary["executables"][1]["exports"][0]["symbol"] == (
        "mixed_matmuls_dispatch_1_matmul_2x3x2_f32"
    )
    assert summary["executables"][1]["exports"][0]["workgroup_size"] == [2, 3, 1]


def test_cli_writes_json() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fixture = Path(tmpdir) / "fixture.mlir"
        fixture.write_text(FIXTURE, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SUMMARY), str(fixture)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    output = json.loads(result.stdout)
    assert output["target"] == "gfx1101"
    assert output["executables"][0]["executable"] == "simple_mul_dispatch_0"


def main() -> int:
    test_parse_metadata()
    test_parse_multi_executable_metadata()
    test_cli_writes_json()
    print("iree_metadata_summary_test: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
