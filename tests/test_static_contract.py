"""Checks that run with the Python standard library alone, without torch."""

import ast
import dataclasses
from pathlib import Path
import re
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hexapod_rl import ActorCriticConfig


class ConfigurationTests(unittest.TestCase):
    def test_default_configuration_matches_local_baseline(self):
        config = ActorCriticConfig()
        task = (ROOT / "hexapod.py").read_text(encoding="utf-8")
        yaml = (ROOT / "hexapodPPO.yaml").read_text(encoding="utf-8")
        # Deliberately narrow static assertions, not a substitute YAML parser.
        for key, value in (("numObservations", config.observation_dim), ("numActions", config.action_dim)):
            self.assertRegex(task, re.escape('self.cfg["env"]["' + key + '"]') + r"\s*=\s*" + str(value) + r"\b")
        units = re.search(r"^\s+units:\s*\[([^\]]+)\]", yaml, re.MULTILINE)
        self.assertIsNotNone(units)
        self.assertEqual(tuple(int(x.strip()) for x in units.group(1).split(",")), config.hidden_sizes)
        self.assertRegex(yaml, r"activation:\s+" + config.activation + r"\b")
        self.assertRegex(yaml, r"separate:\s+True")
        self.assertRegex(yaml, r"fixed_sigma:\s+True")
        self.assertRegex(yaml, r"name:\s+continuous_a2c_logstd")
        self.assertRegex(yaml, r"val:\s+0\.")
        self.assertEqual(config.initial_log_std, 0.)

    def test_config_rejects_invalid_dimensions(self):
        for kwargs in ({"action_dim": 0}, {"observation_dim": -1}, {"hidden_sizes": (4, 0)},
                       {"action_dim": True}, {"hidden_sizes": (1.5,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActorCriticConfig(**kwargs)

    def test_config_rejects_invalid_activation_and_std(self):
        for kwargs in ({"activation": "unknown"}, {"initial_log_std": float("nan")},
                       {"initial_log_std": float("inf")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ActorCriticConfig(**kwargs)

    def test_config_is_immutable_and_accepts_hidden_list(self):
        widths = [8, 4]
        config = ActorCriticConfig(hidden_sizes=widths)
        widths.append(2)
        self.assertEqual(config.hidden_sizes, (8, 4))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.action_dim = 12


class StaticContractTests(unittest.TestCase):
    def test_python_files_parse_without_importing_dependencies(self):
        for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
            with self.subTest(path=str(path.relative_to(ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_package_imports_only_stdlib_torch_and_itself(self):
        allowed = {"dataclasses", "math", "typing", "torch", "hexapod_rl"}
        for path in (ROOT / "src" / "hexapod_rl").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [name.name for name in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module]
                for name in names:
                    self.assertIn(name.split(".")[0], allowed, str(path))

    def test_active_urdf_has_18_revolute_joints_and_no_mimic(self):
        root = ET.parse(ROOT / "hexapod2" / "urdf" / "test_spider6.urdf").getroot()
        joints = root.findall("joint")
        self.assertEqual(sum(j.get("type") == "revolute" for j in joints), 18)
        self.assertEqual(sum(j.get("type") == "fixed" for j in joints), 2)
        self.assertEqual(len(root.findall(".//mimic")), 0)

    def test_active_mesh_references_exist(self):
        root = ET.parse(ROOT / "hexapod2" / "urdf" / "test_spider6.urdf").getroot()
        for mesh in root.findall(".//mesh"):
            reference = mesh.get("filename")
            self.assertTrue(reference.startswith("package://test_spider6/"))
            self.assertTrue((ROOT / "hexapod2" / reference[len("package://test_spider6/"):]).is_file())

    def test_reference_revisions_are_documented(self):
        text = (ROOT / "docs" / "reference-implementation.md").read_text(encoding="utf-8")
        self.assertIn("fa1c13cc10158ab42e829a0971a9dc9a4544b3b9", text)
        self.assertIn("aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3", text)


if __name__ == "__main__":
    unittest.main()
