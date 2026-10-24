# Fault experiments

The experiment catalog includes encoder outage, position outage, gyro-bias
injection, asymmetric traction loss, hot windings, loaded grade,
encoder transport latency, low charge and emergency stop.

Severe mission faults latch: emergency request, overcurrent, winding
overtemperature and very low SOC. They request zero motion through the
simulated controller. The mission does not automatically rearm after the
event ends. Component APIs expose healthy-dwell and explicit reset logic for
separate supervised-recovery studies.

Encoder age above 150 ms requests zero motion, and fresh samples clear that
transient inhibit. Electrical back-EMF wheel-rate feedback prevents a stale
encoder value from continuing to apply braking torque after wheel reversal.
Absolute-position age above two seconds also requests a transient stop.
Successful new fixes allow navigation to resume.

A fault journal records changes of reason, including recovery, rather than
one alarm per integration step. Reports retain timestamps and reason strings.

Executable mission tests check that the robot moves before emergency
injection, stops afterward, and stays latched after the event window ends.
The encoder-outage regression checks timeout onset, fresh-data recovery,
zero commands during the outage, resumed motion and positive footprint
clearance. Component tests exercise I-squared-t overload, hysteresis,
debounce, watchdog expiry, nonfinite commands and reset interlocks.

The simulation requests braking through active current control. It does not
model a safety-rated stop, a redundant controller, a power disconnect,
certified braking hardware or communication-fault coverage. Motor electrical
feedback and the battery remain available in the encoder-fault experiment.
Combined independent failures require additional models and experiments.
