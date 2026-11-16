import dataclasses
from pathlib import Path
import re
import unittest

from ppo_test_support import HAS_TORCH  # adds src to path without importing torch when absent
from hexapod_rl.config import PPOConfig


class PPOConfigurationTests(unittest.TestCase):
    def test_defaults_match_original_yaml(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / "hexapodPPO.yaml").read_text(encoding="utf-8")
        c = PPOConfig()
        names = ("horizon_length", "minibatch_size", "mini_epochs", "gamma", "tau", "e_clip",
                 "critic_coef", "entropy_coef", "bounds_loss_coef", "learning_rate", "grad_norm", "kl_threshold")
        for name in names:
            match = re.search(r"^\s+" + name + r":\s*([^ #\n]+)", text, re.MULTILINE)
            self.assertIsNotNone(match, name)
            self.assertEqual(float(match.group(1)), getattr(c, name), name)
        for name in ("mixed_precision", "normalize_input", "normalize_value", "normalize_advantage",
                     "value_bootstrap", "clip_value", "truncate_grads"):
            self.assertRegex(text, name + r":\s+True")
            self.assertTrue(getattr(c, name))
        task = (root / "hexapod.yaml").read_text(encoding="utf-8")
        self.assertRegex(task, r"numEnvs:.*\b256\b")
        self.assertEqual(c.num_actors, 256)
        self.assertEqual(c.batch_size, 12288)
        self.assertEqual(c.lr_schedule, "adaptive")
        self.assertRegex(text, r"scale_value:\s+1\.0")
        self.assertEqual(c.reward_scale, 1.)

    def test_invalid_counts_and_nondivisible_batches(self):
        for kwargs in ({"num_actors": True}, {"horizon_length": 0}, {"mini_epochs": 1.5},
                       {"minibatch_size": 7}, {"minibatch_size": 24576}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PPOConfig(**kwargs)

    def test_invalid_numerical_settings(self):
        for kwargs in ({"gamma": 1.1}, {"tau": -.1}, {"e_clip": 1}, {"learning_rate": 0},
                       {"entropy_coef": -1}, {"critic_coef": float("nan")}, {"grad_norm": float("inf")},
                       {"adam_betas": (.9, 1)}, {"normalize_input": 1}, {"lr_schedule": "linear"},
                       {"min_learning_rate": .1, "max_learning_rate": .01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PPOConfig(**kwargs)

    def test_immutable_config_and_custom_batch(self):
        betas = [.8, .99]
        c = PPOConfig(num_actors=2, horizon_length=4, minibatch_size=4, adam_betas=betas)
        betas[0] = .1
        self.assertEqual(c.adam_betas, (.8, .99))
        self.assertEqual(c.batch_size, 8)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.gamma = .8

    def test_singleton_normalization_requires_explicit_disable(self):
        settings = dict(num_actors=1, horizon_length=1, minibatch_size=1)
        with self.assertRaises(ValueError):
            PPOConfig(**settings)
        c = PPOConfig(**settings, normalize_input=False, normalize_value=False, normalize_advantage=False)
        self.assertEqual(c.batch_size, 1)
