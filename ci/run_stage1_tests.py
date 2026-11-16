"""Run Stage 1 + Stage 2 discovery, failing on missing coverage and failures.

No simulator or test-framework dependency is needed. PyTorch is mandatory for
the main entrypoint so the optional numerical tests cannot silently disappear.
"""

from pathlib import Path
import sys
import unittest

MINIMUM_TESTS = 87  # 32 preserved Stage 1 tests + 55 Stage 2 tests


def run_suite(suite, stream=None):
    """Return a process exit code; keep the result policy testable without torch."""
    if stream is None:
        stream = sys.stderr
    discovered = suite.countTestCases()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(
        f"Stage 1 + Stage 2: discovered={discovered}, executed={result.testsRun}, "
        f"failures={len(result.failures)}, errors={len(result.errors)}, "
        f"skipped={len(result.skipped)}, "
        f"expected_failures={len(result.expectedFailures)}, "
        f"unexpected_successes={len(result.unexpectedSuccesses)}",
        file=stream,
    )
    problems = []
    if discovered < MINIMUM_TESTS:
        problems.append(f"Only {discovered} tests discovered; at least {MINIMUM_TESTS} are required")
    if result.testsRun != discovered:
        problems.append(f"Incomplete execution: ran {result.testsRun} of {discovered} discovered tests")
    if result.skipped:
        problems.append(f"Skipped tests are not allowed: {len(result.skipped)}")
    if result.expectedFailures:
        problems.append(f"Expected failures are not accepted as validation: {len(result.expectedFailures)}")
    if not result.wasSuccessful():
        problems.append("unittest reported failures, errors, or unexpected successes; see tracebacks above")
    for problem in problems:
        print(f"::error::{problem}", file=stream)
    return 1 if problems else 0


def main():
    try:
        import torch
    except Exception as error:
        print(f"::error::PyTorch import failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Running Stage 1 + Stage 2 with PyTorch {torch.__version__}", flush=True)
    tests = Path(__file__).resolve().parents[1] / "tests"
    suite = unittest.TestLoader().discover(str(tests), pattern="test_*.py")
    return run_suite(suite)


if __name__ == "__main__":
    raise SystemExit(main())
