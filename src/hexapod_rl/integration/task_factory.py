"""Load only the installed task and its dependencies, without registry side effects."""

import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys


def initialize_isaac_gym():
    if "torch" in sys.modules and "isaacgym" not in sys.modules:
        raise RuntimeError("Isaac Gym must initialize before torch. Start a fresh process with python train_agent.py")
    try:
        return importlib.import_module("isaacgym")
    except ImportError as error:
        raise ImportError("Isaac Gym Preview 4 is required for live training. Install it separately in a compatible Linux environment; offline tests need no simulator") from error


def load_task_class(package_path):
    """Process-local package paths bypass the broad package/task __init__ files.

    All actual task/base/utils code is loaded unchanged from the selected tree.
    This launcher owns the IsaacGymEnvs namespace; do not mix it with make()/the
    registry in the same process. Nothing in the external checkout is edited.
    """
    package = Path(package_path).resolve()
    if "isaacgym" not in sys.modules:
        raise RuntimeError("Call initialize_isaac_gym before loading the task")
    if any(name == "isaacgymenvs" or name.startswith("isaacgymenvs.") for name in sys.modules):
        raise RuntimeError("IsaacGymEnvs is already imported; use a fresh process for the narrow task loader")
    task_file = package / "tasks/hexapod.py"
    if not task_file.is_file():
        raise FileNotFoundError(f"Missing custom task: {task_file}")
    created = []
    try:
        for name, directory in (("isaacgymenvs", package), ("isaacgymenvs.tasks", package / "tasks")):
            spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            spec.submodule_search_locations = [str(directory)]
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(directory / "__init__.py")
            sys.modules[name] = module
            created.append(name)
        module = importlib.import_module("isaacgymenvs.tasks.hexapod")
        return module.Anymal10, getattr(module, "writer", None)
    except BaseException:
        for name in list(sys.modules):
            if name in created or name.startswith("isaacgymenvs."):
                del sys.modules[name]
        raise


def construct_task(application):
    package, root = application.package_path, application.resolved
    asset = package.parent / "assets/urdf/hexapod2"
    if not (asset / "urdf/test_spider6.urdf").is_file() or not (asset / "meshes").is_dir():
        raise FileNotFoundError(f"Place the unchanged hexapod2 directory at {asset}; assets are not installed automatically")
    task_class, task_writer = load_task_class(package)
    try:
        environment = task_class(cfg=root["task"], rl_device=root["rl_device"],
            sim_device=root["sim_device"], graphics_device_id=root["graphics_device_id"],
            headless=root["headless"], virtual_screen_capture=False, force_render=root["force_render"])
        return environment, task_writer
    except BaseException:
        if task_writer is not None:
            task_writer.close()
        raise
