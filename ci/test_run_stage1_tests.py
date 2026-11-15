"""Exercise CI gates without torch; separate from the 32 Stage 1 tests."""

import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from run_stage1_tests import main, run_suite


class RunnerTests(unittest.TestCase):
    def evaluate(self, first=None, count=32):
        tests = [unittest.FunctionTestCase(lambda: None) for _ in range(count)]
        if first is not None:
            tests[0] = first
        output = io.StringIO()
        code = run_suite(unittest.TestSuite(tests), stream=output)
        return code, output.getvalue()

    def test_complete_success(self):
        code, output = self.evaluate()
        self.assertEqual(code, 0, output)
        self.assertIn("discovered=32, executed=32", output)

    def test_additional_tests_are_allowed(self):
        code, output = self.evaluate(count=33)
        self.assertEqual(code, 0, output)

    def test_too_few_tests_fail(self):
        for count in (0, 31):
            with self.subTest(count=count):
                code, output = self.evaluate(count=count)
                self.assertEqual(code, 1)
                self.assertIn("at least 32 are required", output)

    def test_skip_fails_even_when_unittest_is_successful(self):
        def skip():
            raise unittest.SkipTest("simulated missing dependency")

        code, output = self.evaluate(unittest.FunctionTestCase(skip))
        self.assertEqual(code, 1)
        self.assertIn("Skipped tests are not allowed: 1", output)

    def test_assertion_failure_fails(self):
        def fail():
            raise AssertionError("simulated test defect")

        code, output = self.evaluate(unittest.FunctionTestCase(fail))
        self.assertEqual(code, 1)
        self.assertIn("failures=1", output)

    def test_error_fails(self):
        def error():
            raise RuntimeError("simulated import or runtime error")

        code, output = self.evaluate(unittest.FunctionTestCase(error))
        self.assertEqual(code, 1)
        self.assertIn("errors=1", output)

    def test_incomplete_execution_fails(self):
        class StopEarly(unittest.TestCase):
            def runTest(self):
                pass

            def run(self, result):
                super().run(result)
                result.stop()

        code, output = self.evaluate(StopEarly())
        self.assertEqual(code, 1)
        self.assertIn("Incomplete execution: ran 1 of 32", output)

    def test_expected_failure_is_not_accepted(self):
        class KnownFailure(unittest.TestCase):
            @unittest.expectedFailure
            def runTest(self):
                self.fail("not valid numerical coverage")

        code, output = self.evaluate(KnownFailure())
        self.assertEqual(code, 1)
        self.assertIn("Expected failures are not accepted", output)

    def test_missing_torch_import_fails(self):
        output = io.StringIO()
        with patch.dict(sys.modules, {"torch": None}), contextlib.redirect_stderr(output):
            code = main()
        self.assertEqual(code, 1)
        self.assertIn("PyTorch import failed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
