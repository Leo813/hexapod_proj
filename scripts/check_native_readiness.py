"""Read-only readiness inventory; explicit flags enable isolated native checks."""

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from install_hexapod_task import SOURCE, checked_path, digest, inspect_environment, inspect_installation


MARKER = "HEXAPOD_NATIVE_RESULT="
SDK_FILES = ("docs/index.html", "python/rlgpu_conda_env.yml", "python/setup.py")


def release(value):
    """Only final numeric releases with optional local build tags; unknown stays unknown."""
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\+[A-Za-z0-9_.-]+)?", str(value))
    return tuple(int(v or 0) for v in match.groups()) if match else None


def assess_requirements(inventory):
    project = (SOURCE / "pyproject.toml").read_text(encoding="utf-8")
    python_min = re.search(r'requires-python\s*=\s*">=([0-9.]+)"', project)
    torch_min = re.search(r'"torch>=([0-9.]+)"', project)
    if not python_min or not torch_min:
        raise ValueError("Package requirement syntax changed; readiness parser needs review")
    issues, notices = [], []
    for name, actual, minimum in (("Python", inventory["python_version"], python_min[1]),
                                 ("torch", inventory["distributions"]["torch"], torch_min[1])):
        parsed = release(actual)
        if parsed is None:
            issues.append(f"{name} version missing/unknown: {actual}")
        elif parsed < release(minimum):
            issues.append(f"{name} {actual} does not satisfy project requirement >={minimum}")
    if not inventory["os"].startswith("Linux"):
        issues.append("Native Preview 4 target requires Linux; current OS is not Linux")
    if release(inventory["python_version"]) and release(inventory["python_version"])[:2] != (3, 8):
        issues.append("Python 3.8 is the intersection of project >=3.8 and Preview 4's documented 3.6-3.8 range")
    if 'VERSION_ID="20.04"' not in inventory.get("os_release", "") or 'ID=ubuntu' not in inventory.get("os_release", ""):
        notices.append("Ubuntu 20.04 target not established from OS metadata")
    for name in ("isaacgym",):
        if not isinstance(inventory["package_locations"].get(name), list):
            issues.append(f"Missing discoverable {name} package; unpacked SDK may need separate preparation")
    for name in ("numpy", "omegaconf", "gym", "matplotlib", "tensorboard"):
        if release(inventory["distributions"].get(name)) is None:
            issues.append(f"Required task/configuration dependency missing or version unknown: {name}")
    omega = release(inventory["distributions"].get("omegaconf"))
    if omega is not None and not (release("2.2") <= omega < release("2.4")):
        issues.append("OmegaConf does not satisfy project integration range >=2.2,<2.4")
    cuda = inventory.get("torch_cuda_build")
    if isinstance(cuda, dict) and cuda.get("value") is None:
        issues.append("Installed torch metadata identifies a CPU build")
    elif not isinstance(cuda, dict):
        notices.append("Torch CUDA build unknown; only an explicitly requested native check can verify it")
    notices.append("Metadata compatibility is not native validation; provisional torch 1.10.x conflicts with torch>=1.13")
    return {"package_requirements": {"python": ">=" + python_min[1], "torch": ">=" + torch_min[1]},
            "blockers": issues, "notices": notices}


def sdk_evidence(root):
    if root is None:
        return {"status": "unavailable: no SDK path supplied; contents not inferred"}
    root = checked_path(root)
    files = {}
    for relative in SDK_FILES:
        path = checked_path(root / relative)
        entry = {"path": str(path), "present": path.is_file()}
        if path.is_file():
            entry["sha256"] = digest(path)
            if path.suffix in (".py", ".yml"):
                entry["text"] = path.read_text(encoding="utf-8")[:65536]
                entry["note"] = "Read as text only; not executed or certified as Preview 4"
        files[relative] = entry
    return {"root": str(root), "files": files}


def readiness(target=None, sdk=None, driver_info=False):
    inventory = inspect_environment(target, driver_info)
    report = {"inventory": inventory, **assess_requirements(inventory), "sdk_evidence": sdk_evidence(sdk),
              "native_execution": "not requested", "simulator_compatibility": "unverified"}
    if target is None:
        report["blockers"].append("No IsaacGymEnvs checkout selected for task/hash verification")
    else:
        try:
            plan = inspect_installation(SOURCE, target)
            report["installation"] = {"complete": plan.complete, "revision": plan.revision,
                "warnings": plan.warnings, "errors": plan.errors,
                "files": [{"path": str(dst), "sha256": sha, "status": status} for _, dst, sha, status in plan.files]}
            if not plan.complete:
                report["blockers"].append("Required installed task/configuration/asset files are missing, conflicting or invalid")
        except (OSError, ValueError) as error:
            report["blockers"].append(f"Installation inspection failed: {error}")
    return report


def import_probe(device):
    """Worker only. Import order and CUDA operations have separately named failure stages."""
    stage = "import-order"
    try:
        if "torch" in sys.modules:
            raise RuntimeError("torch already imported: use a fresh worker process")
        stage = "isaacgym-import"
        gym = importlib.import_module("isaacgym")
        stage = "torch-import"
        torch = importlib.import_module("torch")
        stage = "cuda-availability"
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is false")
        stage = "cuda-device"
        index = int(device.split(":")[1])
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {device}; only {torch.cuda.device_count()} CUDA devices visible")
        torch.cuda.set_device(index)
        stage = "cuda-operation"
        value = torch.arange(8, dtype=torch.float32, device=device).square().sum()
        torch.cuda.synchronize(index)
        if value.item() != 140.0:
            raise RuntimeError("CUDA arithmetic returned an unexpected result")
        return {"ok": True, "stage": "imports-and-cuda", "torch": str(torch.__version__),
                "cuda_build": torch.version.cuda, "device": str(value.device),
                "gpu": torch.cuda.get_device_name(index), "isaacgym": str(gym.__file__),
                "simulator_compatibility": "not tested"}
    except Exception as error:
        return {"ok": False, "stage": stage, "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "simulator_compatibility": "not tested"}


def run_worker(device, timeout, target=None, output=None):
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_worker", "--device", device]
    if output is None:
        command.append("--check-imports")
    else:
        command += ["--run-smoke", "--isaacgymenvs", str(target), "--output", str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
            cwd=str(output) if output else None, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    except subprocess.TimeoutExpired as error:
        return {"ok": False, "stage": "worker-timeout", "error": f"Worker exceeded {timeout} seconds and was killed",
                "stdout_tail": str(error.stdout or "")[-12000:], "stderr_tail": str(error.stderr or "")[-12000:]}
    except OSError as error:
        return {"ok": False, "stage": "worker-launch", "error": str(error)}
    lines = [line[len(MARKER):] for line in result.stdout.splitlines() if line.startswith(MARKER)]
    try:
        status = json.loads(lines[-1]) if lines else {"ok": False, "stage": "worker-exit", "error": "No result marker; see native logs"}
        if not isinstance(status, dict) or type(status.get("ok")) is not bool:
            raise ValueError("Malformed worker result")
    except (ValueError, TypeError):
        status = {"ok": False, "stage": "worker-protocol", "error": "Invalid worker JSON"}
    if result.returncode != 0:
        status["ok"] = False
    status.update(returncode=result.returncode, stdout_tail=result.stdout[-12000:], stderr_tail=result.stderr[-12000:])
    if output is not None:
        (output / "native-stdout.log").write_text(result.stdout, encoding="utf-8")
        (output / "native-stderr.log").write_text(result.stderr, encoding="utf-8")
    return status


def reserve_output(path, target):
    path, target = checked_path(path), checked_path(target)
    for protected in (SOURCE, target):
        if path == protected or path in protected.parents or protected in path.parents:
            raise ValueError("Smoke output must be separate from source and external checkout")
    if not path.parent.is_dir():
        raise ValueError("Smoke output parent must already exist")
    path.mkdir()  # exclusive: never reuse a run or overwrite output
    return path


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaacgymenvs", type=Path)
    parser.add_argument("--isaacgym-sdk", type=Path, help="Optional unpacked SDK root; requirements are read as text")
    parser.add_argument("--driver-info", action="store_true", help="Explicit read-only nvidia-smi query")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-imports", action="store_true", help="Authorize fresh-process Isaac Gym -> torch imports and tiny CUDA operation")
    modes.add_argument("--run-smoke", action="store_true", help="Authorize 4-environment, 48-step, one-epoch simulation and checkpoint outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, help="New absolute directory outside source/checkout; required for smoke")
    parser.add_argument("--timeout", type=int, default=180, help="Worker wall-time limit, 1..600 seconds")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(arguments)
    if not re.fullmatch(r"cuda:\d+", args.device) or not 1 <= args.timeout <= 600:
        parser.error("Use cuda:<nonnegative index> and a timeout between 1 and 600 seconds")
    if args.run_smoke and (args.isaacgymenvs is None or args.output is None):
        parser.error("--run-smoke requires --isaacgymenvs and --output")
    if args.output is not None and not args.run_smoke:
        parser.error("--output is only valid with --run-smoke; default inventory writes nothing")
    if args._worker:
        if not (args.check_imports or args.run_smoke):
            parser.error("Worker requires explicit --check-imports or --run-smoke authorization")
        status = import_probe(args.device)
        if status["ok"] and args.run_smoke:
            try:
                from native_smoke import run_smoke
                status = run_smoke(args.isaacgymenvs, args.output, args.device)
            except Exception as error:
                status = {"ok": False, "stage": "smoke-entry", "error": str(error), "traceback": traceback.format_exc()}
        print(MARKER + json.dumps(status, allow_nan=False), flush=True)
        return 0 if status["ok"] else 1
    try:
        report = readiness(args.isaacgymenvs, args.isaacgym_sdk, args.driver_info)
        if args.check_imports:
            # An import probe may investigate a version that fails package policy;
            # it does not execute custom PPO or override those blockers.
            report["native_execution"] = run_worker(args.device, args.timeout)
        elif args.run_smoke:
            if report["blockers"]:
                report["native_execution"] = {"ok": False, "stage": "preflight", "error": "Resolve reported blockers before smoke execution"}
            else:
                output = reserve_output(args.output, args.isaacgymenvs)
                report["native_execution"] = run_worker(args.device, args.timeout, args.isaacgymenvs, output)
                (output / "readiness.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(report, indent=2, allow_nan=False))
        active = report["native_execution"]
        return 1 if report["blockers"] or (isinstance(active, dict) and not active["ok"]) else 0
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "stage": "inspection", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
