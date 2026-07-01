import math
import unittest

from controlpid import PID, ziegler_nichols, step_response_metrics


class TestPID(unittest.TestCase):
    def test_proportional_only_step(self):
        pid = PID(1, 0, 0, setpoint=1.0)
        self.assertTrue(math.isclose(pid.step(0.0, 0.1), 1.0))

    def test_readme_example(self):
        pid = PID(2, 0.5, 0.1, setpoint=1.0)
        self.assertTrue(math.isclose(round(pid.step(0.0, 0.1), 3), 2.05))

    def test_derivative_on_measurement_has_no_setpoint_kick(self):
        # With the measurement held constant, changing the setpoint must not
        # produce a derivative kick: the derivative term stays zero.
        pid = PID(0, 0, 1, setpoint=0.0, n_filter=10)
        pid.step(0.0, 0.01)
        pid.setpoint = 1.0
        self.assertEqual(pid.step(0.0, 0.01), 0.0)

    def test_filtered_derivative_recursion(self):
        # A unit step in the measurement gives -n*kd/(1+n*dt); on a subsequent
        # held sample the filtered term decays by another 1/(1+n*dt).
        pid = PID(0, 0, 1, setpoint=5.0, n_filter=10)
        pid.step(0.0, 0.01)
        first = pid.step(1.0, 0.01)
        self.assertTrue(math.isclose(first, -10.0 / 1.1))
        second = pid.step(1.0, 0.01)
        self.assertTrue(math.isclose(second, (-10.0 / 1.1) / 1.1))

    def test_anti_windup_clamps_output_and_unwinds_integral(self):
        pid = PID(10, 5, 0, setpoint=1.0, out_min=-1.0, out_max=1.0)
        out = None
        for _ in range(20):
            out = pid.step(0.0, 0.1)
        self.assertEqual(out, 1.0)          # clamped to the configured maximum
        self.assertLess(pid._i, 0)          # integral did not run away

    def test_reset_clears_state(self):
        pid = PID(1, 1, 1, setpoint=1.0)
        pid.step(0.0, 0.1)
        pid.reset()
        self.assertEqual(pid._i, 0.0)
        self.assertEqual(pid._d, 0.0)
        self.assertEqual(pid._prevm, 0.0)


class TestZieglerNichols(unittest.TestCase):
    def test_classic(self):
        g = ziegler_nichols(4.0, 2.0)
        self.assertTrue(math.isclose(g["kp"], 2.4))
        self.assertTrue(math.isclose(g["ki"], 1.2 * 4.0 / 2.0))
        self.assertTrue(math.isclose(g["kd"], 0.075 * 4.0 * 2.0))

    def test_pi_and_p(self):
        pi = ziegler_nichols(4.0, 2.0, kind="PI")
        self.assertTrue(math.isclose(pi["kp"], 0.45 * 4.0))
        self.assertTrue(math.isclose(pi["ki"], 0.54 * 4.0 / 2.0))
        self.assertEqual(pi["kd"], 0.0)

        p = ziegler_nichols(4.0, 2.0, kind="P")
        self.assertTrue(math.isclose(p["kp"], 0.5 * 4.0))
        self.assertEqual(p["ki"], 0.0)
        self.assertEqual(p["kd"], 0.0)

    def test_rejects_nonpositive_tu(self):
        with self.assertRaises(ValueError):
            ziegler_nichols(1.0, 0.0)


class TestStepResponseMetrics(unittest.TestCase):
    def test_positive_setpoint(self):
        m = step_response_metrics([0.0, 0.5, 0.95, 1.2, 1.0], 0.1, 1.0)
        self.assertEqual(m["peak"], 1.2)
        self.assertTrue(math.isclose(m["overshoot_pct"], 20.0))

    def test_negative_setpoint(self):
        # The peak is the most-negative sample, giving +20% overshoot of -1.0.
        m = step_response_metrics([-0.5, -1.2, -1.0], 0.1, -1.0)
        self.assertEqual(m["peak"], -1.2)
        self.assertTrue(math.isclose(m["overshoot_pct"], 20.0))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            step_response_metrics([], 0.1, 1.0)


if __name__ == "__main__":
    unittest.main()
