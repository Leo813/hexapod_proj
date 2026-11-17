"""Prepare unchanged task files in an existing checkout; never install dependencies.

Standard-library only, Python 3.8+. Preview is the default. Run --help for modes.
"""

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile


REFERENCE = "aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3"
SOURCE = Path(__file__).resolve().parents[1]
PREREQUISITES = ("isaacgymenvs/tasks/base/vec_task.py", "isaacgymenvs/cfg/config.yaml")
PACKAGE_DIRS = ("isaacgymenvs", "isaacgymenvs/tasks", "isaacgymenvs/tasks/base",
                "isaacgymenvs/cfg", "isaacgymenvs/utils")
MAPPINGS = (("hexapod.py", "isaacgymenvs/tasks/hexapod.py"),
            ("hexapod.yaml", "isaacgymenvs/cfg/task/hexapod.yaml"),
            ("hexapodPPO.yaml", "isaacgymenvs/cfg/train/hexapodPPO.yaml"))
COMPATIBILITY = ("Unresolved runtime conflict: project requires torch>=1.13; the provisional "
                 "Python 3.8 / torch 1.10.x / CUDA 11.x target is not verified. "
                 "File placement does not resolve this conflict or verify GPU simulation.")


class InstallationError(ValueError):
    pass


def checked_path(path):
    """Reject traversal, symlinks and Windows junction/reparse points, including ancestors."""
    path = Path(path).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise InstallationError(f"Use an absolute path without '..': {path}")
    for part in reversed((path,) + tuple(path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise InstallationError(f"Symbolic link/reparse point is not allowed: {part}")
        if part != path and not stat.S_ISDIR(info.st_mode):
            raise InstallationError(f"Parent is not a directory: {part}")
    return path


def within(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise InstallationError(f"Unsafe relative path: {relative}")
    path = checked_path(root / relative)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise InstallationError(f"Path escapes checkout: {path}") from error
    return path


def digest(path):
    path = checked_path(path)
    if not path.is_file():
        raise InstallationError(f"Required regular file missing: {path}")
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_manifest(source):
    source = checked_path(source)
    assets = within(source, "hexapod2")
    if not assets.is_dir():
        raise InstallationError(f"Missing original asset directory: {assets}")
    for relative in ("hexapod2/urdf/test_spider6.urdf",):
        digest(within(source, relative))
    if not within(source, "hexapod2/meshes").is_dir():
        raise InstallationError("Missing original hexapod2/meshes directory")
    pairs = [(within(source, src), Path(dst)) for src, dst in MAPPINGS]
    # Do not recurse through links (including Windows junctions).
    def visit(directory):
        for path in sorted(directory.iterdir()):
            checked_path(path)
            if path.is_dir():
                visit(path)
            elif path.is_file():
                pairs.append((path, Path("assets/urdf/hexapod2") / path.relative_to(assets)))
            else:
                raise InstallationError(f"Unsupported source file type: {path}")
    visit(assets)
    if not any(src.parent == assets / "meshes" for src, _ in pairs):
        raise InstallationError("Original meshes directory contains no files")
    return [(src, dst, digest(src)) for src, dst in pairs]


def git_revision(target):
    """Read only; do not mistake a containing repository for the chosen checkout."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        def git(*args):
            return subprocess.run(["git", "-C", str(target), "rev-parse", *args],
                check=True, capture_output=True, text=True, timeout=10, env=env).stdout.strip()
        top = Path(git("--show-toplevel")).resolve()
        if top != target.resolve():
            return None, "Target is inside another Git repository, not its checkout root."
        revision = git("HEAD")
    except (OSError, subprocess.SubprocessError):
        return None, "Git revision unavailable (not a checkout, Git missing, or Git inspection failed)."
    notice = "" if revision == REFERENCE else f"Revision differs from selected reference {REFERENCE}; compatibility is unverified, not proven incompatible."
    return revision, notice


@dataclass
class Plan:
    source: Path
    target: Path
    files: list = field(default_factory=list)
    directories: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    protected: dict = field(default_factory=dict)
    revision: str = None

    @property
    def complete(self):
        return not self.errors and bool(self.files) and all(row[3] == "identical" for row in self.files)


def inspect_installation(source, target):
    source, target = checked_path(source), checked_path(target)
    # Installing into the source or an enclosing checkout could alter originals.
    if source == target or source in target.parents or target in source.parents:
        raise InstallationError("Source and target directories must not overlap")
    plan = Plan(source, target)
    if not target.is_dir():
        plan.errors.append(f"External checkout root must already exist: {target}")
    for relative in PACKAGE_DIRS:
        path = within(target, relative)
        if not path.is_dir():
            plan.errors.append(f"Missing package directory: {path}")
    for relative in PREREQUISITES:
        path = within(target, relative)
        if not path.is_file():
            plan.errors.append(f"Missing prerequisite file: {path}")
        else:
            plan.protected[relative] = digest(path)
    directories = set()
    for src, relative, sha in source_manifest(source):
        dst = within(target, relative)
        for parent in dst.parents:
            if parent == target:
                break
            directories.add(parent)
        if not dst.exists():
            status = "missing"
        elif dst.is_file() and digest(dst) == sha:
            status = "identical"
        else:
            status = "conflict"
            plan.errors.append(f"Conflicting destination: {dst}")
        plan.files.append((src, dst, sha, status))
    plan.directories = sorted(directories)
    for directory in plan.directories:
        checked_path(directory)
        if directory.exists() and not directory.is_dir():
            plan.errors.append(f"Destination directory blocked by a file: {directory}")
    if target.is_dir():
        plan.revision, notice = git_revision(target)
        if notice:
            plan.warnings.append(notice)
    plan.warnings.append(COMPATIBILITY)
    return plan


def atomic_create(source, destination, expected_hash, root):
    """Publish a complete new file with an exclusive hard link; never replace a file.

    Requires a filesystem supporting same-directory hard links. There is no
    unsafe copy/overwrite fallback. Use a checkout not being changed concurrently.
    """
    checked_path(source)
    within(root, destination.relative_to(root))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(destination.parent), prefix=".hexapod-", delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if digest(temporary) != expected_hash:
            raise InstallationError(f"Source changed while copying: {source}")
        within(root, destination.relative_to(root))
        os.link(str(temporary), str(destination))  # fails if destination appeared
        if digest(destination) != expected_hash:
            raise InstallationError(f"Installed content verification failed: {destination}")
    finally:
        if temporary is not None:
            temporary.unlink()  # only the uniquely created temporary file


def apply_installation(source, target):
    # Fresh full preflight: a prior preview never authorizes stale file contents.
    plan = inspect_installation(source, target)
    if plan.errors:
        raise InstallationError("Installation refused before writing: " + "; ".join(plan.errors))
    for directory in sorted(plan.directories, key=lambda p: len(p.parts)):
        checked_path(directory)
        directory.mkdir(exist_ok=True)
    for src, dst, sha, status in plan.files:
        if status == "missing":
            atomic_create(src, dst, sha, plan.target)
    for relative, sha in plan.protected.items():
        if digest(within(plan.target, relative)) != sha:
            raise InstallationError(f"External prerequisite changed during installation: {relative}")
    verified = inspect_installation(source, target)
    if not verified.complete:
        raise InstallationError("Post-installation verification failed; inspect and resolve reported files")
    return verified


def inspect_environment(target=None, driver_info=False):
    """Metadata and source text only: never import torch, Gym, or their packages."""
    result = {"os": " ".join((platform.system(), platform.release(), platform.version())), "python_executable": sys.executable,
              "python_version": platform.python_version(), "distributions": {},
              "package_locations": {}, "torch_cuda_build": "unknown (no active import)",
              "driver": "not queried; use --driver-info for a read-only nvidia-smi query",
              "compatibility": COMPATIBILITY, "native_validation": "not performed"}
    os_release = Path("/etc/os-release")
    if sys.platform.startswith("linux") and os_release.is_file():
        result["os_release"] = os_release.read_text(encoding="utf-8")
    for name in ("torch", "isaacgym", "isaacgymenvs", "numpy", "omegaconf", "gym", "hydra-core",
                 "matplotlib", "tensorboard", "hexapod-rl"):
        try:
            result["distributions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["distributions"][name] = "not found in distribution metadata"
    for name in ("torch", "isaacgym", "isaacgymenvs"):
        spec = importlib.util.find_spec(name)  # top-level names; no package initializer
        locations = list(spec.submodule_search_locations or []) if spec else []
        result["package_locations"][name] = locations or "not found on Python path"
        if name == "torch" and locations:
            version_file = Path(locations[0]) / "version.py"
            try:
                for node in ast.parse(version_file.read_text(encoding="utf-8")).body:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
                    if any(isinstance(t, ast.Name) and t.id == "cuda" for t in targets):
                        result["torch_cuda_build"] = {"value": ast.literal_eval(node.value), "source": str(version_file),
                                                      "verified_by_import": False}
            except (OSError, SyntaxError, ValueError):
                pass
    if target is not None:
        target = checked_path(target)
        result["selected_checkout"] = str(target)
        result["checkout_revision"], result["checkout_notice"] = git_revision(target)
    if driver_info:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            result["driver"] = "nvidia-smi not found"
        else:
            try:
                query = subprocess.run([executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                    check=True, capture_output=True, text=True, timeout=10)
                result["driver"] = query.stdout.strip()
            except (OSError, subprocess.SubprocessError) as error:
                result["driver"] = f"Driver query failed: {error}"
    return result


def show_plan(plan):
    print(f"Source repository: {plan.source}\nTarget checkout: {plan.target}")
    print(f"Git revision: {plan.revision or 'unknown'}")
    for directory in plan.directories:
        print(f"DIRECTORY {'present' if directory.is_dir() else 'required'}: {directory}")
    for src, dst, sha, status in plan.files:
        print(f"{status.upper()}: {src.relative_to(plan.source)} -> {dst.relative_to(plan.target)} SHA256={sha}")
    for message in plan.warnings:
        print(f"WARNING: {message}")
    for message in plan.errors:
        print(f"ERROR: {message}")


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacgymenvs", type=Path, help="Absolute external checkout root (not its package subdirectory)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="Explicitly authorize creating missing task files")
    modes.add_argument("--verify", action="store_true", help="Read-only hash verification; missing/conflicting files fail")
    modes.add_argument("--inspect-environment", action="store_true", help="Read-only package metadata inventory; no tensor/simulator imports")
    parser.add_argument("--driver-info", action="store_true", help="With --inspect-environment, request read-only nvidia-smi information")
    args = parser.parse_args(arguments)
    if args.driver_info and not args.inspect_environment:
        parser.error("--driver-info requires --inspect-environment")
    if not args.inspect_environment and args.isaacgymenvs is None:
        parser.error("--isaacgymenvs is required for preview, apply and verification")
    try:
        if args.inspect_environment:
            print(json.dumps(inspect_environment(args.isaacgymenvs, args.driver_info), indent=2))
            return 0
        plan = inspect_installation(SOURCE, args.isaacgymenvs)
        show_plan(plan)
        if plan.errors:
            return 1
        if args.apply:
            plan = apply_installation(SOURCE, args.isaacgymenvs)
            print("Installation verified: required file contents match. GPU simulation NOT verified.")
        elif args.verify:
            print("File placement verified. GPU simulation NOT verified." if plan.complete else "Verification failed: files are missing.")
            return 0 if plan.complete else 1
        else:
            print("Preview only: no files written. Use --apply to create missing files.")
        return 0
    except (OSError, InstallationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
