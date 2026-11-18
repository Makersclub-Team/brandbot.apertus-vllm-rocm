#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect an Apertus ROCm runtime; this does not load model weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ARCHITECTURE = "Apertus1p5ForConditionalGeneration"
TRANSFORMERS_CLASS = "Apertus1p5VisionTokenizerModel"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true",
                        help="Also require an accessible GPU and check a small FP32 matmul on device 0.")
    parser.add_argument("--output", type=Path, default=Path("runtime-report.json"))
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "apertus_rocm_runtime_check",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Package imports, model registration, and optional GPU tensor operation only.",
        "versions": {"python": platform.python_version()},
        "require_gpu": args.require_gpu,
        "checks": [],
        "not_run": ["model_weight_loading", "text_generation", "image_input", "audio_input",
                    "tool_calling", "performance", "multi_gpu_execution"],
    }

    def run(name: str, operation: Callable[[], dict[str, Any]]) -> bool:
        try:
            details = operation()
        except Exception as exc:
            report["checks"].append({"name": name, "status": "failed",
                                     "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            return False
        report["checks"].append({"name": name, "status": "passed", "details": details})
        return True

    def not_run(name: str, reason: str) -> None:
        report["checks"].append({"name": name, "status": "not_run", "reason": reason})

    modules: dict[str, Any] = {}
    for package in ("torch", "vllm", "transformers"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = None

        def load_package(name: str = package) -> dict[str, Any]:
            module = importlib.import_module(name)
            modules[name] = module
            version = getattr(module, "__version__", None)
            if version is not None:
                report["versions"][name] = str(version)
            return {"module": name}

        run(f"import.{package}", load_package)

    torch = modules.get("torch")
    rocm_ok = False
    if torch is not None:
        report["versions"]["torch_rocm_build"] = getattr(torch.version, "hip", None)
        report["versions"]["torch_cuda_build"] = getattr(torch.version, "cuda", None)

        def check_rocm() -> dict[str, Any]:
            hip = report["versions"]["torch_rocm_build"]
            cuda = report["versions"]["torch_cuda_build"]
            if not hip or cuda is not None:
                raise RuntimeError("Expected a ROCm PyTorch build: torch.version.hip must be set "
                                   "and torch.version.cuda must be None.")
            return {"hip": hip, "cuda": cuda}

        rocm_ok = run("torch.rocm_build", check_rocm)
    else:
        not_run("torch.rocm_build", "PyTorch could not be imported.")

    if "vllm" in modules:
        def check_registry() -> dict[str, Any]:
            registry_module = importlib.import_module("vllm.model_executor.models.registry")
            architectures = registry_module.ModelRegistry.get_supported_archs()
            if ARCHITECTURE not in architectures:
                raise RuntimeError(f"ModelRegistry does not register {ARCHITECTURE}.")
            return {"architecture": ARCHITECTURE}

        run("vllm.model_registry", check_registry)

        def check_model_class() -> dict[str, Any]:
            module = importlib.import_module("vllm.model_executor.models.apertus_mm")
            model_class = getattr(module, ARCHITECTURE)
            if not inspect.isclass(model_class):
                raise TypeError(f"{ARCHITECTURE} is not a class.")
            return {"class": ARCHITECTURE, "module": model_class.__module__}

        run("vllm.apertus_model_import", check_model_class)

        def check_registered_parser(module_name: str, manager_name: str,
                                    resolver_name: str) -> dict[str, Any]:
            module = importlib.import_module(module_name)
            manager = getattr(module, manager_name)
            parser_class = getattr(manager, resolver_name)("apertus")
            if not inspect.isclass(parser_class):
                raise TypeError("The registered apertus parser is not a class.")
            return {"registered_name": "apertus", "class": parser_class.__name__,
                    "module": parser_class.__module__}

        def check_audio() -> dict[str, Any]:
            # vLLM swaps missing audio packages for placeholders that only fail
            # when a request carries audio; fail here instead.
            import io

            import numpy as np
            import soundfile
            import soxr

            audio_module = importlib.import_module("vllm.multimodal.audio")
            placeholder = importlib.import_module("vllm.utils.import_utils").PlaceholderModule
            missing = [name for name in ("av", "soxr")
                       if isinstance(getattr(audio_module, name, None), placeholder)]
            if missing:
                raise RuntimeError(f"vllm.multimodal.audio is missing {missing}; install vllm[audio].")
            importlib.import_module("scipy.signal")
            buffer = io.BytesIO()
            soundfile.write(buffer, np.zeros(24000, dtype=np.float32), 24000, format="WAV")
            buffer.seek(0)
            samples, rate = soundfile.read(buffer, dtype="float32")
            resampled = soxr.resample(samples, rate, 16000)
            if len(resampled) != 16000:
                raise RuntimeError(f"Resampling 1 s of audio gave {len(resampled)} samples, expected 16000.")
            versions = {name: importlib.metadata.version(name)
                        for name in ("av", "scipy", "soundfile", "soxr", "mistral_common")}
            torchaudio_present = not isinstance(getattr(audio_module, "torchaudio", None), placeholder)
            return {"packages": versions, "torchaudio_present": torchaudio_present,
                    "wav_decode_and_resample": "24000 Hz -> 16000 Hz"}

        run("vllm.audio_support", check_audio)

        run("vllm.apertus_tool_parser", lambda: check_registered_parser(
            "vllm.tool_parsers", "ToolParserManager", "get_tool_parser"))
        run("vllm.apertus_reasoning_parser", lambda: check_registered_parser(
            "vllm.reasoning", "ReasoningParserManager", "get_reasoning_parser"))
    else:
        not_run("vllm.model_registry", "vLLM could not be imported.")
        not_run("vllm.apertus_model_import", "vLLM could not be imported.")
        not_run("vllm.audio_support", "vLLM could not be imported.")
        not_run("vllm.apertus_tool_parser", "vLLM could not be imported.")
        not_run("vllm.apertus_reasoning_parser", "vLLM could not be imported.")

    if "transformers" in modules:
        def check_transformers_class() -> dict[str, Any]:
            model_class = getattr(modules["transformers"], TRANSFORMERS_CLASS)
            if not inspect.isclass(model_class):
                raise TypeError(f"{TRANSFORMERS_CLASS} is not a class.")
            return {"class": TRANSFORMERS_CLASS, "module": model_class.__module__}

        run("transformers.apertus_vision_class_import", check_transformers_class)
    else:
        not_run("transformers.apertus_vision_class_import", "Transformers could not be imported.")

    if not args.require_gpu:
        not_run("gpu.fp32_matmul", "--require-gpu was not supplied; no GPU execution was checked.")
    elif not rocm_ok:
        not_run("gpu.fp32_matmul", "A valid ROCm PyTorch build was not confirmed.")
    else:
        def check_gpu() -> dict[str, Any]:
            # ROCm PyTorch deliberately uses the torch.cuda compatibility API.
            if not torch.cuda.is_available():
                raise RuntimeError("ROCm PyTorch cannot access a GPU through torch.cuda.")
            device_count = torch.cuda.device_count()
            properties = torch.cuda.get_device_properties(0)
            matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda:0")
            actual = (matrix @ matrix).cpu()
            torch.cuda.synchronize(0)
            expected = torch.tensor([[7.0, 10.0], [15.0, 22.0]], dtype=torch.float32)
            if not torch.allclose(actual, expected, rtol=0.0, atol=1e-6):
                raise RuntimeError("The FP32 matrix multiplication result differed from the expected result.")
            return {"visible_device_count": device_count, "tested_device_index": 0,
                    "device_name": torch.cuda.get_device_name(0),
                    "gcn_arch_name": getattr(properties, "gcnArchName", None),
                    "operation": "2x2 FP32 matrix multiplication", "absolute_tolerance": 1e-6}

        run("gpu.fp32_matmul", check_gpu)

    report["status"] = "failed" if any(c["status"] == "failed" for c in report["checks"]) else "passed"
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    except OSError as exc:
        print(serialized, end="")
        print(f"Cannot write JSON report: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(serialized, end="")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
