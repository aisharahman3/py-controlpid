# controlpid

PID with anti-windup + derivative filtering and Ziegler-Nichols tuning.

BSD-3-Clause license.

## Example

```python
import controlpid

pid = controlpid.PID(2, 0.5, 0.1, setpoint=1.0)
print(round(pid.step(0.0, 0.1), 3))  # 2.05
```
