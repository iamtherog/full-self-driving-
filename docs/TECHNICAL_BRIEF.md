# autodrive: technical brief

A driver-assistance software stack (lane keeping, adaptive cruise, traffic
lights, emergency braking, driver override, fault handling) driving a
simulated 2026 Toyota Corolla, with an offline attention monitor that helps
the driver stay engaged. **Simulation only.** It connects to no vehicle, and
the README explains why it must not be put on one.

Every number below comes from the code in this repository and can be
reproduced with the commands at the end.

## Architecture

```
world ─▶ sensors (20 Hz, noisy) ─▶ perception ─▶ planner ─▶ controllers (100 Hz) ─▶ safety supervisor ─▶ vehicle
                                                                                                   │
                                     read-only snapshot after every step ─▶ attention monitor ──▶ advisories (logged only)
```

Design rules, each enforced in code:

- **The planner never sees ground truth**, only noisy sensors run through a
  Kalman-filtered lead tracker and lane estimator.
- **Every longitudinal rule proposes an acceleration and the most cautious
  wins** (cruise, curve, car following, traffic light), so each rule can be
  reasoned about and tested alone.
- **The safety supervisor has the last word and assumes everything above it
  can be wrong.** It enforces a command envelope (+2.0 / −3.5 m/s², 5 m/s³ jerk,
  3 m/s² lateral), hands control back the instant the driver brakes, raises
  TAKE CONTROL on stale perception or lane departure, and runs emergency
  braking whether or not the system is engaged.
- **Observers cannot act.** The attention monitor gets frozen snapshots of what
  the system knows and returns advisories that the loop records and never reads.

## Vehicle models

- **Kinematic bicycle** (default): valid at the moderate lateral acceleration
  the planner allows.
- **Dynamic single-track model with Fiala brush tires** (`--dynamic`): lateral
  tire slip and saturation (Rajamani 2012; Pacejka 2012). Validated against
  steady-state theory (understeer-gradient equation within 2 %; lateral
  acceleration bounded by μg at full lock).

All six scenarios pass on both, with peak lane error 3-11 cm (kinematic) and
4-15 cm (dynamic).

**Against a 2026 Corolla:** dimensions, mass and turning circle match published
figures (within 1-2 %). The simulated car is slightly slower to 60 mph (9.2 s vs
7.8-8.9 s tested), so it never flatters the results. The braking limit and tire
grip are not verified and were chosen conservatively. Feature by feature, the
stack covers the vehicle-following, lane-centering and lane-departure parts of
Toyota Safety Sense 3.0 in simulation. It does not cover pedestrian, cyclist or
intersection detection, road signs or high beams.

## Attention monitor

An interpolated modified Kneser-Ney n-gram model (Chen & Goodman 1999) of the
event stream of 1,500 randomized nominal drives. Each 0.5 s window becomes one
token: mode, planner behavior, braking, time-to-collision and lane offset, each
taken as the worst value in the window. Online it scores each event's surprisal
and flags unlikely or never-seen events.

- **The model decides when; templates decide what.** Messages are filled with
  measured values, never generated.
- **It defers.** It only voices what the system measured and nobody has told
  the driver (closing, hard braking, lane offset, stale perception). It is
  silent during and just after supervisor alerts and while the driver is
  steering.
- **Model order** is chosen by Wald's SPRT with one trial per independent drive
  (per-token trials are correlated and would void the error bounds).

| Measure | Result |
| --- | --- |
| hard_brake scenario | advisories 1.65 s and 0.65 s before emergency braking |
| highway, city, override, sensor failure | no advisories |
| held-out nominal driving (3.7 h) | 0.27 advisories/h (95 % CI 0.007-1.5) |
| review-log flags on the same | 2.7/h (95 % CI 1.3-5.0), above the 1/h calibration target |

The design follows the Lean N-gram Generator by Roger Feeley Lussier. This is
an independent implementation from the published algorithms.

## Evidence

| What | How it is checked |
| --- | --- |
| Correctness | 105 tests: unit, closed-loop, and property tests (e.g. every Kneser-Ney distribution sums to 1 for seen and unseen contexts) |
| Tests that matter | Mutation testing: each part of the planner and vehicle-model fixes was disabled in turn, and a test fails every time |
| Isolation | Every scenario driven with and without the monitor; logs must be identical |
| Reproducibility | Shipped model retrained from recorded seeds reproduces its SHA-256 bit for bit (`attention verify`) |
| Statistics | SPRT error rates checked against Wald's bounds by simulation; Poisson intervals against the chi-square identity |
| No regressions | The six original scenarios are bit-identical after every change to the stack |
| Offline | Training and monitoring run with sockets disabled |
| Code quality | ruff and `mypy --strict` clean; CI on Python 3.10-3.12 for every push |

## Defects found and fixed

| Defect | How it was found | Fix |
| --- | --- | --- |
| Jerk reached 6.7 m/s³ against a 5 m/s³ limit (fault entry, re-engage) | Re-running the original scorecard | Supervisor rate-limits acceleration itself (AEB exempt) |
| Car could run a red light at a yellow | Nominal-corpus filter rejected a drive | Brake build-up allowance, latched stop/go decisions, go-commitment lapses at low speed; generator uses ITE yellow timing |
| 1-ulp probability difference after save/load | Round-trip test | Order-independent sums (`math.fsum`), canonical count order |
| Steering cap throttled cornering on a slipping car (lane-departure fault, 5.6 m) | Dynamic vehicle model | Cap includes the understeer term |
| Dead reckoning drifted 0.92 m through a sensor dropout | Dynamic vehicle model | Uses the yaw-rate sensor instead of the no-slip formula |
| Lane error in curves on a slipping car | Dynamic vehicle model | Understeer and rear-slip feedforward in the lateral controller |

The red-light fix was checked on seeds never used in debugging. With realistic
(ITE) signal timing, neither the old nor the new planner ran a red light in
3,000 drives. The remaining failures need yellows shorter than the ITE interval,
which leave a car that can neither stop nor clear, and no planner can fix that
without knowing the signal timing.

## Known limits

- Single lane: no lane changes, pedestrians or cross traffic.
- Perception is simulated noise on ground truth, not camera images.
- The dynamic model has lateral slip only (no longitudinal slip, load transfer
  or suspension). Braking limit, grip, cornering stiffness and yaw inertia are
  not verified against published Corolla data.
- The attention monitor has only seen simulated drives; its false-alarm rate on
  real traffic is unknown. It is not a certified driver monitoring system.
- Nothing is validated against real vehicle data.

## Reproduce

```
pip install -r requirements-dev.txt
python -m pytest                          # 104 tests (~40 s)
python -m pytest -m slow                  # bit-for-bit retrain (~4 min)
python -m autodrive                       # scorecard, kinematic vehicle
python -m autodrive --dynamic             # scorecard, tire-slip vehicle
python -m autodrive.attention report      # attention monitor on the named scenarios
python -m autodrive.attention verify      # artifact integrity and reproducibility
ruff check autodrive tests && python -m mypy --strict autodrive
```

## References

- R. Rajamani, *Vehicle Dynamics and Control*, 2nd ed., Springer, 2012.
- H. B. Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., Elsevier, 2012.
- P. Polack et al., The kinematic bicycle model: a consistent model for planning
  feasible trajectories for autonomous vehicles? IEEE IV, 2017.
- J. Kong et al., Kinematic and dynamic vehicle models for autonomous driving
  control design. IEEE IV, 2015.
- R. Kneser, H. Ney, Improved backing-off for m-gram language modeling. ICASSP, 1995.
- S. F. Chen, J. Goodman, An empirical study of smoothing techniques for
  language modeling. Computer Speech and Language 13(4), 1999.
- A. Wald, Sequential tests of statistical hypotheses. Annals of Mathematical
  Statistics 16(2), 1945.
