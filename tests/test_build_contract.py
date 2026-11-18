# SPDX-License-Identifier: Apache-2.0
"""Check offline behavior and version-pinning requirements of the build wrapper."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build.py"


class BuildContractTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args],
                              text=True, capture_output=True)

    def test_floating_base_is_rejected(self):
        result = self.invoke("--base-image", "rocm/vllm-dev:base", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("immutable image reference", result.stderr)

    def test_dry_run_is_offline_and_does_not_claim_build(self):
        # Synthetic digest is only a parser fixture and is never used to pull.
        before = (ROOT / ".build").exists()
        result = self.invoke("--base-image", "example.invalid/base@sha256:" + "a" * 64,
                             "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "plan-only-not-built")
        self.assertEqual(data["gpu_validation"], "not_run")
        self.assertEqual((ROOT / ".build").exists(), before)
        self.assertEqual(data["base_image_source"], "external")
        self.assertEqual(len(data["commands"]), 2)

    def test_default_uses_the_pinned_prebuilt_base(self):
        versions = json.loads((ROOT / "versions.json").read_text())
        result = self.invoke("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["base_image_source"], "versions.json")
        self.assertEqual(data["base_image"], versions["rocm_base_image"])
        self.assertRegex(versions["rocm_base_image"], r"@sha256:[0-9a-f]{64}$")
        self.assertEqual(len(data["commands"]), 2)
        self.assertIn("BASE_IMAGE=" + versions["rocm_base_image"], data["commands"][0])

    def test_build_base_builds_from_the_pinned_source(self):
        result = self.invoke("--build-base", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["base_image_source"], "built-from-pinned-source")
        self.assertEqual(len(data["commands"]), 3)
        base, engine, _ = data["commands"]
        self.assertIn("docker/Dockerfile.rocm_base", base)
        self.assertIn("PYTORCH_ROCM_ARCH=gfx950", base)
        self.assertIn("BASE_IMAGE=" + data["base_image"], engine)

    def test_aiter_gets_only_architectures_it_supports(self):
        result = self.invoke("--build-base", "--gpu-arch", "gfx90a;gfx942", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        base = json.loads(result.stdout)["commands"][0]
        self.assertIn("'PYTORCH_ROCM_ARCH=gfx90a;gfx942'", base)
        self.assertIn("AITER_ROCM_ARCH=gfx942", base)
        self.assertNotIn("AITER_ROCM_ARCH=gfx90a", base)

    def test_pins_and_known_conflict_are_consistent(self):
        versions = json.loads((ROOT / "versions.json").read_text())
        self.assertEqual(versions["vllm_branch"], "apertus-1-5")
        self.assertEqual(versions["rocm_sdk_version"], "10.0.0")
        conflicts = versions["known_dependency_conflicts"]
        self.assertEqual([c["requires"].split(">")[0].split("<")[0] for c in conflicts],
                         ["transformers", "tokenizers"])
        # The Dockerfile tolerates exactly the documented Transformers version.
        dockerfile = (ROOT / "Dockerfile").read_text()
        allowed = versions["transformers_version"].replace(".", "\\.")
        self.assertIn("but you have transformers " + allowed, dockerfile)
        self.assertIn("tokenizers>=0\\.23\\.2, but you have tokenizers", dockerfile)
        for package, spec in versions["transformers_runtime_pins"].items():
            self.assertIn(f'"{package}{spec}"', dockerfile)

    def test_invalid_architecture_is_rejected(self):
        result = self.invoke("--base-image", "example.invalid/base@sha256:" + "a" * 64,
                             "--gpu-arch", "unrecognized", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gfx architecture", result.stderr)


if __name__ == "__main__":
    unittest.main()
