"""Model-version contract tests without PyTorch or downloaded weights."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import export_uma_state as exporter
import unittest

ROOT = Path(__file__).resolve().parents[2]


class UmaModelVersionTests(unittest.TestCase):
    def test_converter_exposes_pinned_121(self):
        result = subprocess.run([sys.executable, str(ROOT / "converters/convert.py"),
                                 "info", "uma-s-1p2p1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout)
        self.assertEqual(info["checkpoint"]["sha256"],
                         "b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce")
        self.assertEqual(info["checkpoint"]["size_bytes"], 2333393615)
        self.assertIn("uma-s-1p2p1", info["exporter"]["extra_args"])

    def test_exporter_accepts_model_selection(self):
        result = subprocess.run([sys.executable, str(ROOT / "tools/reference/export_uma_state.py"),
                                 "--help"], capture_output=True, text=True)
        self.assertIn("--model", result.stdout)

    def test_manifest_preserves_version_identity(self):
        for model, sha, size, version in [
            ("uma-s-1p2", "ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b", 2333393167, "2.20.0"),
            ("uma-s-1p2p1", "b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce", 2333393615, "2.22.0"),
        ]:
            with tempfile.TemporaryDirectory() as directory:
                exporter.write_manifest(Path(directory), [], True, model_key=model)
                text = (Path(directory) / "manifest.toml").read_text()
                self.assertIn(f'selected_model = "{model}"', text)
                self.assertIn(f'state_schema = "{model}-state-dict-v1"', text)
                self.assertIn(f'artifact_sha256 = "{sha}"', text)
                self.assertIn(f'artifact_size_bytes = {size}', text)
                self.assertIn(f'reference_version = "{version}"', text)

    def test_checkpoint_config_rejects_wrong_model_id(self):
        for model, model_id in [("uma-s-1p2", "UMA-S-1.2"), ("uma-s-1p2p1", "UMA-S-1.2.1")]:
            exporter.validate_model_config({"model_id": model_id}, model)
            with self.assertRaises(ValueError):
                exporter.validate_model_config({"model_id": "UMA-S-other"}, model)


if __name__ == "__main__":
    unittest.main()
