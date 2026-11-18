"""Entry-point and ignore-rule regressions, without native dependencies."""

import ast
import contextlib
import io
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hexapod_rl import launcher


class EntryPointTests(unittest.TestCase):
    def test_renamed_entrypoint_only_imports_existing_launcher_and_stdlib(self):
        self.assertFalse((ROOT / 'train.py').exists())
        tree = ast.parse((ROOT / 'train_agent.py').read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
                if node.module == 'hexapod_rl.launcher':
                    self.assertEqual([alias.name for alias in node.names], ['main'])
        self.assertEqual(imports, ['pathlib', 'sys', 'hexapod_rl.launcher'])

    def test_entrypoint_preserves_arguments_import_order_and_exit_codes(self):
        for error, expected in ((None, 0), (ValueError('invalid'), 1),
                                (KeyboardInterrupt(), 130)):
            events = []
            application = object()

            def configure(arguments):
                self.assertEqual(arguments, ['headless=True', 'num_envs=64'])
                events.append('configuration')
                return application

            def run(config):
                self.assertIs(config, application)
                events.append('application')
                if error is not None:
                    raise error

            with self.subTest(expected=expected), \
                    patch.object(launcher, 'load_configuration', side_effect=configure), \
                    patch.object(launcher, 'initialize_isaac_gym', side_effect=lambda: events.append('isaacgym')), \
                    patch.object(launcher, 'run_application', side_effect=run), \
                    patch.object(sys, 'argv', ['train_agent.py', 'headless=True', 'num_envs=64']), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(ROOT / 'train_agent.py'), run_name='__main__')
            self.assertEqual(stopped.exception.code, expected)
            self.assertEqual(events, ['configuration', 'isaacgym', 'application'])

    def ignored(self, paths):
        result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin', '-z'],
                                input=('\0'.join(paths) + '\0').encode(), cwd=ROOT,
                                capture_output=True)
        self.assertIn(result.returncode, (0, 1), result.stderr)
        return set(result.stdout.decode().rstrip('\0').split('\0')) if result.stdout else set()

    def test_generated_outputs_are_ignored(self):
        paths = ['runs/experiment/checkpoints/last.pt', 'events.out.tfevents.123',
                 'summaries/events.out.tfevents.123', 'build/lib/package.py',
                 'dist/package.whl', 'src/hexapod_rl.egg-info/PKG-INFO',
                 '.mypy_cache/cache.json', '.ruff_cache/cache']
        self.assertEqual(self.ignored(paths), set(paths))

    def test_source_assets_and_checkpoint_fixtures_are_not_ignored(self):
        paths = ['train_agent.py', 'hexapod.py', 'hexapod.yaml', 'hexapodPPO.yaml',
                 'hexapodPPO_LSTM.yaml', 'Paper_yuhaotsai.pdf',
                 'hexapod2/urdf/test_spider6.urdf', 'hexapod2/meshes/link.STL',
                 'tests/fixtures/reference.pt', 'tests/fixtures/reference.pth',
                 'src/hexapod_rl/ppo.py']
        self.assertEqual(self.ignored(paths), set())
