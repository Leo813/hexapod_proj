import io
import math
import unittest

from ppo_test_support import HAS_TORCH

if HAS_TORCH:
    import torch
    from hexapod_rl.normalization import RunningMeanStd


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class NormalizationTests(unittest.TestCase):
    def test_initial_state_precision_and_epsilon(self):
        norm = RunningMeanStd(1).eval()
        self.assertEqual(norm.count.item(), 1)
        self.assertEqual(norm.running_mean.dtype, torch.float64)
        self.assertEqual(norm.running_var.dtype, torch.float64)
        torch.testing.assert_close(norm(torch.ones(1, 1)), torch.tensor([[1 / math.sqrt(1.00001)]]))

    def test_known_sample_variance_and_sequential_updates(self):
        norm = RunningMeanStd(1)
        norm(torch.tensor([[1.], [3.]]))
        self.assertAlmostEqual(norm.running_mean.item(), 4 / 3)
        self.assertAlmostEqual(norm.running_var.item(), 23 / 9)
        self.assertEqual(norm.count.item(), 3)
        norm(torch.tensor([[5.], [7.]]))
        self.assertAlmostEqual(norm.running_mean.item(), 16 / 5)
        self.assertAlmostEqual(norm.running_var.item(), 189 / 25)
        self.assertEqual(norm.count.item(), 5)

    def test_normalize_denormalize_and_eval_freezes(self):
        norm = RunningMeanStd(2, clip=None)
        x = torch.tensor([[1., -2.], [3., 4.]])
        norm(x)
        norm.eval()
        saved = {k: v.clone() for k, v in norm.state_dict().items()}
        torch.testing.assert_close(norm(norm(x), unnorm=True), x)
        for k, v in norm.state_dict().items():
            torch.testing.assert_close(v, saved[k], rtol=0, atol=0)

    def test_clip_applies_before_denormalizing(self):
        norm = RunningMeanStd(1, epsilon=0.25, clip=2).eval()
        norm.running_mean.fill_(3)
        norm.running_var.fill_(4)
        torch.testing.assert_close(norm(torch.tensor([[100.], [-100.]])), torch.tensor([[2.], [-2.]]))
        expected = torch.tensor([[3 + 2 * math.sqrt(4.25)], [3 - 2 * math.sqrt(4.25)]])
        torch.testing.assert_close(norm(torch.tensor([[100.], [-100.]]), unnorm=True), expected)

    def test_state_dict_round_trip(self):
        norm = RunningMeanStd(1)
        norm(torch.tensor([[1.], [3.]]))
        stream = io.BytesIO()
        torch.save(norm.state_dict(), stream)
        stream.seek(0)
        restored = RunningMeanStd(1).eval()
        restored.load_state_dict(torch.load(stream, map_location="cpu"))
        norm.eval()
        torch.testing.assert_close(restored(torch.tensor([[2.]])), norm(torch.tensor([[2.]])), rtol=0, atol=0)
        self.assertEqual(restored.count.item(), 3)

    def test_constant_batch_finite_and_singleton_training_rejected(self):
        norm = RunningMeanStd(1)
        self.assertTrue(torch.isfinite(norm(torch.zeros(2, 1))).all())
        self.assertAlmostEqual(norm.running_var.item(), 1 / 3)
        with self.assertRaises(ValueError):
            norm(torch.zeros(1, 1))
        self.assertEqual(norm.count.item(), 3)
        norm.eval()
        self.assertTrue(torch.isfinite(norm(torch.zeros(1, 1))).all())

    def test_training_unnorm_also_updates(self):
        norm = RunningMeanStd(1)
        result = norm(torch.tensor([[1.], [3.]]), unnorm=True)
        self.assertEqual(norm.count.item(), 3)
        expected = torch.tensor([[4 / 3 + math.sqrt(23 / 9 + 1e-5)],
                                 [4 / 3 + 3 * math.sqrt(23 / 9 + 1e-5)]])
        torch.testing.assert_close(result, expected)

    def test_invalid_configuration_and_shapes(self):
        for args in ((0,), (1, 0), (1, float("nan")), (1, 1e-5, -2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                RunningMeanStd(*args)
        with self.assertRaises(ValueError):
            RunningMeanStd(1)(torch.ones(2))
