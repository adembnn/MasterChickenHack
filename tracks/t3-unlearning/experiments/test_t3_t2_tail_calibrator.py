#!/usr/bin/env python3
"""CPU-only synthetic regression tests for t3_t2_tail_calibrator; no EAC files."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("cal", Path(__file__).with_name("t3_t2_tail_calibrator.py"))
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class TailTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(20261122)
        n, c = 512, 100
        self.y = rng.integers(0, c, size=n)
        x = rng.lognormal(0, .8, size=(n, c))
        x[np.arange(n), self.y] = rng.uniform(.0001, .6, n)
        self.q = x / x.sum(axis=1, keepdims=True)

    def test_beta0_is_identity_and_true_class_CE_is_invariant(self):
        rng = np.random.default_rng(900)
        orig_t2 = cal.at_temperature_two(self.q)
        tail = cal.conditional_tail(orig_t2, self.y)
        z = rng.normal(0, 2, size=self.q.shape)
        teacher = cal.conditional_tail(cal.at_temperature_two(cal.probabilities(z, "logits")), self.y)
        ce = -np.log(self.q[np.arange(len(self.y)), self.y])
        for beta in (0., .2, .4, .6, 1.):
            q1, q2 = cal.blend_target(self.q, self.y, tail, teacher, beta)
            self.assertTrue(np.isfinite(q1).all() and np.isfinite(q2).all())
            self.assertLess(np.max(abs(q1.sum(1)-1)), 1e-10)
            self.assertLess(np.max(abs(q2.sum(1)-1)), 1e-10)
            self.assertLess(np.max(abs(-np.log(q1[np.arange(len(self.y)),self.y])-ce)), 1e-10)
            if beta == 0.:
                self.assertLess(np.max(abs(q1-self.q)), 1e-11)

    def test_arithmetic_T2_teacher_mean_is_not_mean_logits(self):
        rng = np.random.default_rng(777)
        z = rng.normal(0, 5, size=(4, 300, 100))
        p2 = np.mean([cal.at_temperature_two(cal.probabilities(t, "logits")) for t in z], axis=0)
        wrong = cal.at_temperature_two(cal.probabilities(z.mean(axis=0), "logits"))
        self.assertGreater(float(np.mean(abs(p2-wrong))), .001)

    def test_loo_cv_remains_finite(self):
        rng = np.random.default_rng(11)
        p2 = cal.at_temperature_two(self.q)
        teachers = []
        for i in range(5):
            z = np.log(np.maximum(p2, 1e-12))*2 + rng.normal(0, .1, size=p2.shape)
            teachers.append(cal.at_temperature_two(cal.probabilities(z, "logits")))
        scores = cal.screen_cv(self.q, self.y, teachers, (0., .2, .4, .6))
        self.assertTrue(all(np.isfinite(s["heldout_tail_kl"]) for s in scores.values()))


if __name__ == "__main__":
    unittest.main()
