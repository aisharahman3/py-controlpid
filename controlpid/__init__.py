from .tuning import ziegler_nichols

class PID:
    """Positional-form PID with output clamping, integral anti-windup and a
    first-order derivative filter (N ~ 10 is typical)."""

    def __init__(self, kp, ki, kd, setpoint=0.0,
                 out_min=float("-inf"), out_max=float("inf"), n_filter=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.setpoint = setpoint
        self.out_min, self.out_max = out_min, out_max
        self.n = n_filter
        self._i = 0.0
        self._d = 0.0
        self._prevm = 0.0

    def reset(self):
        self._i = self._d = self._prevm = 0.0

    def step(self, measurement, dt):
        err = self.setpoint - measurement
        self._i += self.ki * err * dt
        # Derivative on measurement (not error) so a setpoint change doesn't
        # produce a kick; low-pass filtered with time constant 1/N.
        self._d = (self._d - self.n * self.kd * (measurement - self._prevm)) / (1 + self.n * dt)
        self._prevm = measurement
        out = self.kp * err + self._i + self._d
        clamped = min(max(out, self.out_min), self.out_max)
        if clamped != out:            # anti-windup: stop integrating when saturated
            self._i -= (out - clamped)
        return clamped


def step_response_metrics(samples, dt, setpoint):
    if not samples:
        raise ValueError("samples must be non-empty")
    # The peak is the sample that travels furthest past the setpoint in the
    # direction of the step, so this works for negative setpoints too.
    if setpoint >= 0:
        peak = max(samples)
    else:
        peak = min(samples)
    overshoot = (peak - setpoint) / setpoint * 100 if setpoint else 0.0
    # Rise time measured between 10% and 90% of the setpoint.
    t10 = t90 = None
    for i, v in enumerate(samples):
        reached_10 = v >= 0.1 * setpoint if setpoint >= 0 else v <= 0.1 * setpoint
        reached_90 = v >= 0.9 * setpoint if setpoint >= 0 else v <= 0.9 * setpoint
        if t10 is None and reached_10:
            t10 = i * dt
        if reached_90:
            t90 = i * dt
            break
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    return {"overshoot_pct": overshoot, "rise_time": rise, "peak": peak}
