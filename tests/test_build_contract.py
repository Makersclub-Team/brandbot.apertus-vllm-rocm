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
        self.assertEqual(len(data["commands"]), 2)

    def test_invalid_architecture_is_rejected(self):
        result = self.invoke("--base-image", "example.invalid/base@sha256:" + "a" * 64,
                             "--gpu-arch", "unrecognized", "--dry-run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gfx architecture", result.stderr)


if __name__ == "__main__":
    unittest.main()
