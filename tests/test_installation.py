"""Stage 4A: real file I/O in temporary checkouts, no simulator or torch required."""

import ast
import contextlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import install_hexapod_task as installer


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


@contextlib.contextmanager
def link_or_windows_simulation(link, target, directory=True):
    """Real symlinks on CI; reparse detection double if Windows denies creation."""
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if os.name != "nt" or getattr(error, "winerror", None) != 1314:
            raise
        # No skipped tests on Windows without the CreateSymbolicLink privilege.
        link.mkdir() if directory else link.touch()
        original = Path.lstat
        def lstat(path, *args, **kwargs):
            if path == link:
                return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", lstat):
            yield
    else:
        yield


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source, self.target = self.base / "source", self.base / "IsaacGymEnvs"
        self.source.mkdir()
        self.target.mkdir()
        for src, _ in installer.MAPPINGS:
            (self.source / src).write_bytes((src + "\r\noriginal\n").encode())
        for relative, content in (("hexapod2/urdf/test_spider6.urdf", b"original URDF"),
                                  ("hexapod2/meshes/robot.STL", bytes(range(256)) + b"\x00\xff")):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        for relative in installer.PACKAGE_DIRS:
            (self.target / relative).mkdir(parents=True, exist_ok=True)
        for relative in installer.PREREQUISITES:
            (self.target / relative).write_bytes(b"historical external prerequisite\n")
        self.git = patch.object(installer, "git_revision", return_value=(installer.REFERENCE, ""))
        self.git.start()
        self.addCleanup(self.git.stop)

    def cli(self, *args):
        output = io.StringIO()
        with patch.object(installer, "SOURCE", self.source), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = installer.main(["--isaacgymenvs", str(self.target), *args])
        return code, output.getvalue()

    def test_default_preview_makes_no_changes(self):
        before = snapshot(self.base)
        code, output = self.cli()
        self.assertEqual(code, 0, output)
        self.assertIn("Preview only", output)
        self.assertIn("MISSING:", output)
        self.assertIn("SHA256=", output)
        self.assertEqual(snapshot(self.base), before)
        self.assertFalse((self.target / "assets").exists())

    def test_apply_preserves_real_source_manifest_and_all_binary_meshes(self):
        # The production manifest, not just a made-up subset, exercises the layout.
        source_before = {src: sha for src, _, sha in installer.source_manifest(ROOT)}
        protected = snapshot(self.target)
        plan = installer.apply_installation(ROOT, self.target)
        self.assertTrue(plan.complete)
        expected = {Path(dst) for _, dst, _ in installer.source_manifest(ROOT)}
        self.assertEqual(len(expected), 35)
        self.assertEqual(sum(p.suffix.lower() == ".stl" for p in expected), 25)
        for src, dst, sha, status in plan.files:
            self.assertEqual(status, "identical")
            self.assertEqual(dst.read_bytes(), src.read_bytes())
            self.assertEqual(installer.digest(dst), sha)
            self.assertEqual(installer.digest(src), source_before[src])
        for relative, value in protected.items():
            self.assertEqual(snapshot(self.target)[relative], value)
        urdf = self.target / "assets/urdf/hexapod2/urdf/test_spider6.urdf"
        for mesh in ET.parse(urdf).getroot().findall(".//mesh"):
            reference = mesh.get("filename")
            self.assertTrue(reference.startswith("package://test_spider6/"))
            relative = reference[len("package://test_spider6/"):]
            self.assertTrue((self.target / "assets/urdf/hexapod2" / relative).is_file())
        self.assertFalse((self.target / "isaacgymenvs/cfg/train/hexapodPPO_LSTM.yaml").exists())
        self.assertFalse(any("Paper" in str(p) for p in expected))

    def test_repeated_installation_does_not_touch_existing_files(self):
        installer.apply_installation(self.source, self.target)
        before = snapshot(self.base)
        with patch.object(installer, "atomic_create") as create:
            self.assertTrue(installer.apply_installation(self.source, self.target).complete)
        create.assert_not_called()
        self.assertEqual(snapshot(self.base), before)

    def test_identical_file_is_preserved_during_partial_installation(self):
        dst = self.target / installer.MAPPINGS[0][1]
        dst.write_bytes((self.source / "hexapod.py").read_bytes())
        timestamp = dst.stat().st_mtime_ns
        plan = installer.inspect_installation(self.source, self.target)
        self.assertEqual(plan.files[0][3], "identical")
        installer.apply_installation(self.source, self.target)
        self.assertEqual(dst.stat().st_mtime_ns, timestamp)

    def test_conflict_prevents_every_write_and_preserves_unrelated_files(self):
        conflict = self.target / installer.MAPPINGS[0][1]
        conflict.write_bytes(b"local changes")
        (self.target / "unrelated.txt").write_bytes(b"keep me")
        before = snapshot(self.base)
        code, output = self.cli("--apply")
        self.assertEqual(code, 1)
        self.assertIn("CONFLICT:", output)
        self.assertEqual(snapshot(self.base), before)
        self.assertFalse((self.target / "assets").exists())

    def test_missing_prerequisites_are_reported_without_writes(self):
        for relative in installer.PREREQUISITES:
            path = self.target / relative
            contents = path.read_bytes()
            path.unlink()
            before = snapshot(self.base)
            with self.subTest(relative=relative):
                code, output = self.cli("--apply")
                self.assertEqual(code, 1)
                self.assertIn("Missing prerequisite", output)
                self.assertIn(str(path), output)
                self.assertEqual(snapshot(self.base), before)
            path.write_bytes(contents)

    def test_unsafe_relative_traversal_and_overlapping_paths_rejected(self):
        for target in (Path("relative"), self.target / ".." / "elsewhere", self.source,
                       self.source / "nested", self.source.parent):
            with self.subTest(target=target), self.assertRaises(installer.InstallationError):
                installer.inspect_installation(self.source, target)
        for relative in ("../outside", str(self.base / "outside")):
            with self.assertRaises(installer.InstallationError):
                installer.within(self.target, relative)

    def test_destination_symlink_escape_is_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        with link_or_windows_simulation(self.target / "assets", outside):
            with self.assertRaisesRegex(installer.InstallationError, "link/reparse"):
                installer.apply_installation(self.source, self.target)
        self.assertEqual(list(outside.iterdir()), [])

    def test_source_and_destination_file_symlinks_are_rejected(self):
        outside = self.base / "outside.bin"
        outside.write_bytes(b"external data")
        with link_or_windows_simulation(self.target / installer.MAPPINGS[0][1], outside, False):
            with self.assertRaisesRegex(installer.InstallationError, "link/reparse"):
                installer.inspect_installation(self.source, self.target)
        # Independent source link, even though it would be copied unchanged.
        with link_or_windows_simulation(self.source / "hexapod2/meshes/extra.STL", outside, False):
            with self.assertRaisesRegex(installer.InstallationError, "link/reparse"):
                installer.source_manifest(self.source)
        self.assertEqual(outside.read_bytes(), b"external data")

    def test_windows_junction_attribute_is_rejected(self):
        original = Path.lstat
        def lstat(path, *args, **kwargs):
            if path == self.target:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return original(path, *args, **kwargs)
        with patch.object(Path, "lstat", lstat), self.assertRaisesRegex(installer.InstallationError, "reparse"):
            installer.inspect_installation(self.source, self.target)

    def test_verify_reports_modified_and_missing_files(self):
        self.assertEqual(self.cli("--verify")[0], 1)
        installer.apply_installation(self.source, self.target)
        self.assertEqual(self.cli("--verify")[0], 0)
        dst = self.target / installer.MAPPINGS[0][1]
        dst.write_bytes(b"changed")
        self.assertEqual(self.cli("--verify")[0], 1)
        dst.unlink()
        code, output = self.cli("--verify")
        self.assertEqual(code, 1)
        self.assertIn("MISSING:", output)

    def test_directory_file_collision_is_rejected(self):
        (self.target / "assets").write_bytes(b"not a directory")
        before = snapshot(self.base)
        self.assertEqual(self.cli("--apply")[0], 1)
        self.assertEqual(snapshot(self.base), before)

    def test_source_change_before_publication_leaves_destination_absent(self):
        source = self.source / "hexapod.py"
        expected = installer.digest(source)
        source.write_bytes(b"changed after plan")
        dst = self.target / "new.py"
        with self.assertRaisesRegex(installer.InstallationError, "Source changed"):
            installer.atomic_create(source, dst, expected, self.target)
        self.assertFalse(dst.exists())
        self.assertEqual(list(self.target.glob(".hexapod-*")), [])

    def test_concurrent_file_creation_cannot_be_overwritten(self):
        source = self.source / "hexapod.py"
        dst = self.target / "new.py"
        native_link = os.link
        def race(temp, destination):
            Path(destination).write_bytes(b"concurrently created")
            native_link(temp, destination)
        with patch.object(installer.os, "link", side_effect=race), self.assertRaises(FileExistsError):
            installer.atomic_create(source, dst, installer.digest(source), self.target)
        self.assertEqual(dst.read_bytes(), b"concurrently created")
        self.assertEqual(list(self.target.glob(".hexapod-*")), [])

    def test_failed_publication_never_falls_back_to_overwriting(self):
        before = snapshot(self.base)
        with patch.object(installer.os, "link", side_effect=OSError("hard links unavailable")):
            self.assertEqual(self.cli("--apply")[0], 1)
        self.assertEqual(snapshot(self.base), before)
        self.assertEqual(list(self.target.rglob(".hexapod-*")), [])

    def test_readonly_git_revision_and_mismatch_warning(self):
        # Mock command responses, not Git availability; no new repository needed.
        self.git.stop()
        with patch.object(installer.subprocess, "run", side_effect=[
                SimpleNamespace(stdout=str(self.target)), SimpleNamespace(stdout="a" * 40)]) as run:
            revision, warning = installer.git_revision(self.target)
        self.assertEqual(revision, "a" * 40)
        self.assertIn("not proven incompatible", warning)
        for call in run.call_args_list:
            self.assertEqual(call.args[0][:4], ["git", "-C", str(self.target), "rev-parse"])
            self.assertEqual(call.kwargs["env"]["GIT_OPTIONAL_LOCKS"], "0")

    def test_unknown_git_revision_is_informative_not_a_write_blocker(self):
        with patch.object(installer, "git_revision", return_value=(None, "Git revision unavailable")):
            plan = installer.inspect_installation(self.source, self.target)
        self.assertFalse(plan.errors)
        self.assertIn("Git revision unavailable", plan.warnings)

    def test_environment_inspection_never_imports_tensor_or_simulator_packages(self):
        script = ROOT / "scripts/install_hexapod_task.py"
        code = """
import importlib.abc, runpy, sys
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'isaacgym', 'isaacgymenvs'):
            raise AssertionError('Active import attempted: ' + fullname)
# Replace discovery with inert specs, then reject any actual import.
import importlib.util
importlib.util.find_spec = lambda name: None
sys.meta_path.insert(0, Reject())
sys.argv = [sys.argv[1], '--inspect-environment']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(script)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout)
        self.assertIn("torch>=1.13", info["compatibility"])
        for expected in ("Python 3.8", "PyTorch 1.13.1 CUDA 11.7", "Isaac Gym Preview 4",
                         "native binary compatibility remains unverified",
                         "passing metadata checks do not establish native readiness"):
            self.assertIn(expected, info["compatibility"])
        self.assertNotIn("1.10.x", info["compatibility"])
        self.assertEqual(info["native_validation"], "not performed")

    def test_cuda_build_is_read_as_text_not_executed(self):
        package = self.base / "torch"
        package.mkdir()
        (package / "version.py").write_text("raise RuntimeError('must not execute')\ncuda: str = '11.7'\n", encoding="utf-8")
        spec = SimpleNamespace(submodule_search_locations=[str(package)])
        with patch.object(installer.importlib.util, "find_spec", side_effect=lambda name: spec if name == "torch" else None):
            info = installer.inspect_environment()
        self.assertEqual(info["torch_cuda_build"]["value"], "11.7")
        self.assertFalse(info["torch_cuda_build"]["verified_by_import"])

    def test_driver_query_is_only_explicit_and_handles_missing_executable(self):
        with patch.object(installer.subprocess, "run") as run:
            installer.inspect_environment()
        run.assert_not_called()
        with patch.object(installer.shutil, "which", return_value=None):
            self.assertEqual(installer.inspect_environment(driver_info=True)["driver"], "nvidia-smi not found")

    def test_no_dependency_install_commands_are_executed(self):
        # Git is mocked separately; no shell/pip/conda/download execution remains.
        with patch.object(installer.subprocess, "run", side_effect=AssertionError("unexpected command")):
            self.assertEqual(self.cli()[0], 0)
            self.assertEqual(self.cli("--apply")[0], 0)
            self.assertEqual(self.cli("--verify")[0], 0)
            installer.inspect_environment()
        tree = ast.parse(Path(installer.__file__).read_text(encoding="utf-8"))
        imports = {n.name for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
        self.assertFalse(imports & {"torch", "isaacgym", "isaacgymenvs", "requests", "urllib"})

    def test_empty_mesh_source_is_rejected(self):
        (self.source / "hexapod2/meshes/robot.STL").unlink()
        before = snapshot(self.base)
        with self.assertRaisesRegex(installer.InstallationError, "meshes directory contains no files"):
            installer.apply_installation(self.source, self.target)
        self.assertEqual(snapshot(self.base), before)
