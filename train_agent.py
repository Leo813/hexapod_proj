"""Custom PPO entry point. Isaac Gym must initialize before importing torch."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hexapod_rl.launcher import main


if __name__ == "__main__":
    raise SystemExit(main())
