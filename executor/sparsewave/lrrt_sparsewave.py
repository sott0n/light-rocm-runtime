import json
from pathlib import Path

import torch

_APPLICATIONS = {}


def register_application(name, loader):
    if name in _APPLICATIONS:
        raise RuntimeError(f"duplicate SparseWave application: {name}")
    _APPLICATIONS[name] = loader


class Executor:
    def __init__(self, implementation):
        self._implementation = implementation

    @classmethod
    def load(cls, manifest: Path | str, **configuration) -> "Executor":
        manifest = Path(manifest).resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"SparseWave manifest not found: {manifest}")

        document = json.loads(manifest.read_text())
        application = document.get("sparsewave", {}).get("application")
        if application is None:
            kernel_names = [
                kernel.get("name") for kernel in document.get("kernels", [])
            ]
            if kernel_names == ["spmm"]:
                application = "spmm"
        loader = _APPLICATIONS.get(application)
        if loader is None:
            raise ValueError(f"unsupported SparseWave application: {application}")
        return cls(loader(manifest, document, configuration))

    def __call__(self, *inputs: torch.Tensor):
        return self._implementation(*inputs)


import lrrt_sparsewave_applications  # noqa: E402, F401
