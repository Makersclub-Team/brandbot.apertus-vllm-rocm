#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the pinned Swiss vLLM ROCm engine and a thin Apertus runtime layer.

Experimental recipe: no GPU or container build result is implied by this script.
By default the ROCm base is AMD's prebuilt rocm/vllm-dev:base, pinned by digest in
versions.json; its labels match the pinned source's docker/Dockerfile.rocm_base.
--build-base builds that Dockerfile instead (hours), and --base-image takes any
other base by immutable digest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA40 = re.compile(r"^[0-9a-f]{40}$")
IMAGE_DIGEST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}$")
ARCH = re.compile(r"^gfx[0-9a-z]+(?:;gfx[0-9a-z]+)*$")
AITER_ARCHES = ("gfx942", "gfx950")


def run(argv: list[str], *, capture: bool = False) -> str:
    print("+ " + shlex.join(argv), flush=True)
    result = subprocess.run(argv, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def load_versions() -> dict:
    versions = json.loads((ROOT / "versions.json").read_text())
    for field in ("vllm_commit", "transformers_commit"):
        if not SHA40.fullmatch(versions[field]):
            raise ValueError(f"{field} must be a full commit SHA")
    return versions


def prepare_checkout(destination: Path, repository: str, commit: str) -> None:
    if not destination.exists():
        run(["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)])
    elif not (destination / ".git").is_dir():
        raise RuntimeError(f"Refusing to overwrite non-Git directory: {destination}")
    else:
        origin = run(["git", "-C", str(destination), "remote", "get-url", "origin"], capture=True)
        if origin != repository:
            raise RuntimeError("Existing source checkout has a different origin")
        dirty = run(["git", "-C", str(destination), "status", "--porcelain"], capture=True)
        if dirty:
            raise RuntimeError("Source checkout has local edits; preserve or remove them before building")
    run(["git", "-C", str(destination), "fetch", "origin", commit])
    run(["git", "-C", str(destination), "checkout", "--detach", commit])
    actual = run(["git", "-C", str(destination), "rev-parse", "HEAD"], capture=True)
    if actual != commit:
        raise RuntimeError("Source checkout does not match the pinned commit")


def main() -> int:
    versions = load_versions()
    parser = argparse.ArgumentParser(description=__doc__)
    base = parser.add_mutually_exclusive_group()
    base.add_argument("--base-image",
                      help="ROCm base as repository@sha256:<digest> "
                           "(default: rocm_base_image in versions.json)")
    base.add_argument("--build-base", action="store_true",
                      help="Build the base from the pinned source's docker/Dockerfile.rocm_base "
                           "instead of using a prebuilt one (compiles Triton, AITER, flash-attention)")
    parser.add_argument("--tag", default="apertus-vllm-rocm:experimental")
    parser.add_argument("--gpu-arch", default=versions["default_gpu_arch"])
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned commands without network, Docker, or filesystem changes")
    args = parser.parse_args()
    if not args.build_base and args.base_image is None:
        args.base_image = versions["rocm_base_image"]
    if args.base_image is not None and not IMAGE_DIGEST.fullmatch(args.base_image):
        parser.error("--base-image requires an immutable image reference ending @sha256:<64 lowercase hex digits>")
    if not ARCH.fullmatch(args.gpu_arch):
        parser.error("--gpu-arch must be a gfx architecture or semicolon-separated gfx architectures")
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    if not args.tag or args.tag.startswith("-") or any(c.isspace() for c in args.tag):
        parser.error("--tag must be a nonempty image tag without whitespace")

    work = ROOT / ".build"
    source = work / "vllm"
    engine_tag = f"apertus-rocm-engine:{versions['vllm_commit'][:12]}"
    built_base_tag = f"apertus-rocm-base:{versions['vllm_commit'][:12]}"
    base_image = args.base_image or built_base_tag
    # The base compiles PyTorch extensions (Triton, AITER, flash-attention) for the
    # target. AITER supports only gfx942 and gfx950, so it gets the overlap.
    aiter_arch = ";".join(a for a in args.gpu_arch.split(";") if a in AITER_ARCHES) or "gfx942;gfx950"
    base_command = None if args.base_image else [
        "docker", "buildx", "build", "--builder", "default", "--load", "--platform", "linux/amd64",
        "--file", str(source / "docker/Dockerfile.rocm_base"),
        "--build-arg", f"PYTORCH_ROCM_ARCH={args.gpu_arch}",
        "--build-arg", f"AITER_ROCM_ARCH={aiter_arch}",
        "--tag", built_base_tag, str(source),
    ]
    engine_command = [
        "docker", "buildx", "build", "--builder", "default", "--load", "--platform", "linux/amd64",
        "--file", str(source / "docker/Dockerfile.rocm"),
        "--target", "vllm-openai",
        "--build-arg", f"BASE_IMAGE={base_image}",
        "--build-arg", f"ARG_PYTORCH_ROCM_ARCH={args.gpu_arch}",
        "--build-arg", f"max_jobs={args.max_jobs}",
        "--build-arg", "NIC_BACKEND=none",
        "--tag", engine_tag, str(source),
    ]
    runtime_command = [
        "docker", "buildx", "build", "--builder", "default", "--load", "--platform", "linux/amd64",
        "--file", str(ROOT / "Dockerfile"),
        "--build-arg", f"APERTUS_ENGINE_IMAGE={engine_tag}",
        "--build-arg", f"APERTUS_VLLM_COMMIT={versions['vllm_commit']}",
        "--build-arg", f"APERTUS_TRANSFORMERS_COMMIT={versions['transformers_commit']}",
        "--build-arg", f"APERTUS_TRANSFORMERS_REPOSITORY={versions['transformers_repository']}",
        "--tag", args.tag, str(ROOT),
    ]
    if args.dry_run:
        print(json.dumps({
            "status": "plan-only-not-built", "versions": versions,
            "source_checkout": str(source),
            "source_repository": versions["vllm_repository"],
            "source_commit": versions["vllm_commit"],
            "base_image": base_image,
            "base_image_source": ("built-from-pinned-source" if args.build_base else
                                  "versions.json" if args.base_image == versions["rocm_base_image"] else "external"),
            "commands": [shlex.join(c) for c in (base_command, engine_command, runtime_command) if c],
            "gpu_validation": "not_run",
        }, indent=2))
        return 0

    for binary in ("git", "docker"):
        if shutil.which(binary) is None:
            parser.error(f"Required executable is missing: {binary}")
    # Use Docker's default builder: it can read the local image between stages.
    # A remote/container builder may try to pull the intermediate local tag.
    run(["docker", "buildx", "inspect", "default"])
    work.mkdir(exist_ok=True)
    prepare_checkout(source, versions["vllm_repository"], versions["vllm_commit"])
    if base_command:
        run(base_command)
    run(engine_command)
    run(runtime_command)
    image_id = run(["docker", "image", "inspect", args.tag, "--format", "{{.Id}}"], capture=True)
    report = {
        "schema_version": 1,
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "versions": versions,
        "base_image": base_image,
        "base_image_source": ("built-from-pinned-source" if args.build_base else
                                  "versions.json" if args.base_image == versions["rocm_base_image"] else "external"),
        "gpu_arch": args.gpu_arch,
        "image_tag": args.tag,
        "local_image_id": image_id,
        "image_built": True,
        "gpu_validation": "not_run",
        "note": "A local image ID is not a registry manifest digest. Record the latter after an intentional push.",
    }
    output = work / "build-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Built local image; GPU validation remains outstanding. Report: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Build failed: {exc}") from exc
