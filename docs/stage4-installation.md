# Stage 4A: safe task preparation and environment inspection

Branch: `feat/stage4-safe-installation`. Starting commit:
`b5dc27a020b82c4c93907a56e6f4c250fbddb1f9` (Stage 3.5B merge).
The working tree was clean; local `main` and `origin/main` both identified that
commit. No fetch, dependency installation, commit, push or publication was
performed for this implementation.

`scripts/install_hexapod_task.py` prepares the **original task files** inside
an existing IsaacGymEnvs checkout. It is not an Isaac Gym installer. It uses
only the Python standard library, supports Python 3.8 syntax, and does not
import torch, Isaac Gym, IsaacGymEnvs or the custom PPO package. No network
operation, pip/Conda command or task-registry edit is part of the utility.

## Runtime assumptions and the unresolved version conflict

| Component | Training target / status |
| --- | --- |
| OS | Ubuntu 20.04 |
| Simulator | NVIDIA Isaac Gym Preview 4 |
| Python | 3.8, provisional |
| PyTorch | 1.10.x, provisional; **conflicts with current `torch>=1.13` package requirement** |
| CUDA | Compatible 11.x build/runtime, provisional |
| Physics and learning | GPU PhysX and GPU PyTorch PPO |
| Parallel environments | 256 |
| IsaacGymEnvs reference | `aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3` |

The Python/PyTorch/CUDA versions are assumptions, not a recovered original
environment lockfile. The selected IsaacGymEnvs revision is a reconstruction
reference, not proof of the exact research checkout. The utility warns about
the dependency conflict on every plan and environment report. It does not
change the requirement, downgrade a working installation, or certify another
revision as incompatible merely because its hash differs.

The existing Ubuntu 22.04 / Python 3.10 / torch 1.13.1+cpu / NumPy 1.24.4 /
OmegaConf 2.3.0 CI stack validates software offline. It is not the simulator
installation recipe. Python 3.8 / torch 1.10.x, CUDA and native Preview 4
execution have not been validated by Stage 4A. The checkpoint old-API trust
path does not establish whole-package support for torch 1.10.

## Inspect without importing the runtime

Run from this repository, using the Python executable whose environment you
want to inventory (using a different Python inspects a different environment):

```bash
python -B scripts/install_hexapod_task.py --inspect-environment
python -B scripts/install_hexapod_task.py --inspect-environment --isaacgymenvs /absolute/path/to/IsaacGymEnvs
```

The JSON report includes OS/Python, package distribution versions and top-level
package locations, and the selected checkout's revision when supplied.
NumPy, OmegaConf, gym, Hydra, matplotlib and TensorBoard are included. The
original task imports matplotlib and creates a TensorBoard writer even when
the custom trainer's optional logging is off; these remain external runtime
prerequisites. Missing distribution metadata is not proof that an unpacked SDK
is absent, and finding a package is not proof that its binary can load.

If available, the torch CUDA build string is read from a literal assignment in
`torch/version.py` with the AST parser, without executing that file. It is
labeled unverified by import; a missing/nonliteral assignment stays unknown.
This reports a build string, not driver compatibility or CUDA availability.
The normal mode does not run a GPU query. For an explicit read-only driver
inventory, add:

```bash
python -B scripts/install_hexapod_task.py --inspect-environment --driver-info
```

This runs only `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`
when available, with a timeout. It does not initialize the simulator or run a
tensor computation. There is no active-import validation mode in Stage 4A.
Future native checks must import Isaac Gym before torch, as required by the
existing launcher.

## Prepare the external checkout

Supply an existing checkout **root**, not its `isaacgymenvs` subdirectory.
The root must be an absolute path without `..`, outside the source repository
and not an ancestor of it. Expected pre-existing paths are:

```text
IsaacGymEnvs/
  isaacgymenvs/
    tasks/base/vec_task.py
    cfg/config.yaml
    utils/
```

The package, tasks/base and cfg directories must exist. The `assets` directory
may be absent: the checkout root is its intended existing parent. The utility
checks the required structure and hashes protected prerequisite files; it
does not verify every transitive import of VecTask. Use the already selected
SDK and checkout. Missing prerequisites must be prepared deliberately outside
this utility; it will not clone, fetch, switch, reset or upgrade that checkout.

For an actual Git checkout, only `git rev-parse --show-toplevel` and `HEAD` are
used, with optional Git locks disabled. A containing parent repository is not
reported as the selected checkout. Unknown revisions or a mismatch produce
informative warnings, while missing files or unsafe paths block installation.
An unknown revision does not block copying into an otherwise valid unpacked
checkout. Local modifications to base code are not certified compatible by
matching the HEAD hash.

## Preview, apply and verify

```bash
# Default: read-only plan. No directories or files are created.
python -B scripts/install_hexapod_task.py --isaacgymenvs /absolute/path/to/IsaacGymEnvs

# Explicit authorization to create only missing files.
python -B scripts/install_hexapod_task.py --isaacgymenvs /absolute/path/to/IsaacGymEnvs --apply

# Read-only verification. Missing or differing required files cause failure.
python -B scripts/install_hexapod_task.py --isaacgymenvs /absolute/path/to/IsaacGymEnvs --verify
```

The source repository is detected relative to the script, independent of the
working directory. Preview lists source/target roots, revision, required
directories, each file's SHA-256, and `MISSING`, `IDENTICAL` or `CONFLICT`
status. Missing prerequisites and unsafe structures produce errors. Exit 0
means a valid preview (missing installable files allowed), completed apply,
or complete verification, depending on the selected mode. Exit 1 means a
filesystem/installation/verification failure; invalid CLI syntax exits 2.
Environment inventory exit 0 means a report was produced, not runtime approval.

The actual source manifest currently has **35 files**: three original task/
configuration files and all 32 files in `hexapod2`, including 25 STL meshes:

```text
IsaacGymEnvs/
  assets/urdf/hexapod2/           # whole original hexapod2 tree
    urdf/test_spider6.urdf
    meshes/...
    config/..., launch/..., package.xml, CMakeLists.txt, urdf/*.csv
  isaacgymenvs/
    tasks/hexapod.py
    cfg/config.yaml             # EXISTING: never installed or overwritten
    cfg/task/hexapod.yaml
    cfg/train/hexapodPPO.yaml
```

This preserves the original `hexapod.py` lookup of `../../assets` and
`urdf/hexapod2/urdf/test_spider6.urdf`. Mesh URI strings stay unchanged.
No research paper, repository documentation or LSTM configuration is copied.
No files are generated from URDFs or meshes. File content is verified before
publication and again in the post-installation manifest. Ownership and
timestamps of original source files are not copied as installation metadata.

## Conflicts, idempotency and filesystem guarantees

- Matching destination files are left untouched, including their modification
  times. A repeated successful apply performs no file copies.
- A differing file or file/directory collision blocks the entire planned copy
  before writes. There is no force flag. Inspect the differences, back up and
  deliberately relocate/remove the conflicting file yourself, or choose a
  separate prepared checkout. The utility never resolves conflicts by deletion.
- Absolute paths, containment and every existing ancestor are checked.
  Symlinks, dangling links and Windows junction/reparse points are rejected
  in source and destination paths, even if they currently point inside the
  checkout. Use physical paths/directories for this workflow.
- Each new file is copied into a unique same-directory temporary file, flushed,
  hashed, and published with an atomic, exclusive hard link. If another file
  appears at the destination, publication fails rather than overwriting it.
  The temporary name is removed; the published file remains. Filesystems must
  support hard links. Failure has no unsafe overwrite/copy fallback.
- The operation is atomic **per file**, not a transaction over all 35 files.
  An I/O failure can leave already verified new files and empty directories;
  existing files are never partially replaced. Correct the failure and preview
  again; verified files will be recognized as identical. Unrelated files are
  neither deleted nor treated as owned by this installer.
- Use a trusted checkout that nobody is changing during installation.
  Ancestor checks are repeated, but this cross-platform utility is not a
  security boundary against malicious concurrent directory/mount replacement.
  It detects prerequisite hash changes without overwriting them or rolling
  back another process's work.

A successful verification means **required file contents and placement match
the source**. It says nothing about URDF import order, PhysX, sensors, driver
compatibility, GPU execution or learning behavior. The root config is required
and left unchanged; the utility does not compare it with a downloaded template
or silently repair device defaults. Review its existing configuration before
native execution.

## Installing the custom package manually

Inventory the existing CUDA stack first. Do not use dependency resolution to
"fix" the unresolved torch 1.10.x versus `torch>=1.13` conflict. Stop and obtain
a separately validated compatibility decision if the installed torch does not
satisfy the project requirement. `--no-deps` does not make that runtime supported.

Only after that decision, in an already prepared environment with compatible
torch, OmegaConf and the declared build prerequisites available, the following
manual commands avoid replacing torch or fetching isolated build dependencies:

```bash
python -m pip install --no-deps --no-build-isolation --editable /absolute/path/to/hexapod_proj
python -m pip check
```

These commands are not executed by the utility or this task. The existing
build requirement is setuptools >=61; appropriate build tooling and optional
dependencies must already be present. Do not install the CPU CI wheel into
the GPU training environment. Do not add RL-Games to the custom PPO package.
Resolve missing dependencies manually against the preserved environment;
this guide does not prescribe a new simulator/CUDA installation.

## Existing launcher and Stage 4B boundary

After file verification and separately approved native compatibility work, the
existing launcher can be used without changes:

```bash
python train.py task=hexapod isaacgymenvs_path=/absolute/path/to/IsaacGymEnvs headless=True
```

The intended baseline is `pipeline=gpu sim_device=cuda:0 rl_device=cuda:0`
with 256 environments. Preserve those settings in the historical root/task
configuration. The launcher imports Isaac Gym before torch and loads the
installed task directly; no registry entry is needed. This command starts
actual simulation/training and **was not run in Stage 4A**.

Stage 4B must validate the real SDK imports, CUDA/driver/runtime combination,
task and mesh loading, native reset/contact behavior, GPU checkpoint resume
and finite training updates. File hash checks and CPU mocks do not establish
these results or scientific reproduction.

## Tests and validation record

Twenty-two standard-library tests in `tests/test_installation.py` use temporary
external checkouts. They cover preview without writes, the real 35-file asset
manifest and binary equality, idempotency/identical-file preservation, conflict
preflight, prerequisites, traversal/overlap/link/reparse rejection, modified or
missing file detection, unrelated-file preservation, failed/concurrent file
publication, Git revision warnings, metadata-only inspection, CUDA version
source parsing, explicit driver queries and absence of dependency installation.

On Windows without symlink creation privileges, the link tests exercise the
reparse rejection branch with a filesystem-stat double; Linux CI creates real
symlinks. No tests are skipped for that platform limitation.

The existing 156 tests remain. The strict CI minimum is now **178**, with no
skips, errors, failures or incomplete execution accepted. The historical CPU
dependency versions and workflow remain unchanged. Local installation tests
initially hit a sandbox denial of hard-link creation in temporary directories;
an approved outside-sandbox run passed all 22. This did not modify a real
Isaac Gym installation. The existing suite was run separately inside the
sandbox: 156 discovered, 36 passed and 120 dependency skips (108 require torch,
12 require OmegaConf), with no failures/errors. Across those two runs the
178 tests therefore yielded 58 passes and 120 skips. All nine separate CI-gate
tests passed; the strict entrypoint correctly rejected missing torch.
All 39 package/test/CI/script files parse with Python 3.8 grammar. Whitespace
checks and protected-source diffs passed. A new successful 178-test GitHub run
with zero skips is still required; these local results are not full CI success.

The read-only local inventory found Windows 11, Python 3.12.14, NumPy 2.3.5,
and no torch, OmegaConf, Isaac Gym or IsaacGymEnvs in the inspected Python's
distribution metadata/path. No driver query was requested, and no external
native checkout path was supplied. Checkout installation was exercised only
against temporary fixtures, including copies of the actual repository assets.
This inventory is not a description of the user's intended Ubuntu GPU machine.
