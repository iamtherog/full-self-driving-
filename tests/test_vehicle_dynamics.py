"""The tire-slip (dynamic) plant: physics against theory, and the stack running on it."""

import math
from dataclasses import replace

import pytest

from autodrive import Config, scenarios
from autodrive.config import rear_slip_gradient, understeer_gradient, use_dynamic_vehicle
from autodrive.safety import SafetySupervisor
from autodrive.sim import run
from autodrive.vehicle import GRAVITY, Vehicle, VehicleCommand, VehicleState, fiala

CFG = Config()
DYN = replace(CFG.vehicle, model="dynamic")


def steady_turn(params, speed, steer, seconds=30.0):
    """Hold speed and road-wheel angle until the car settles; return its state."""
    car = Vehicle(replace(params, max_steer_rate=10.0), VehicleState(speed=speed))
    for _ in range(round(seconds / 0.01)):
        car.step(VehicleCommand(accel=0.0, steer=steer), 0.01)
    return car.state


# --- Tire model ---------------------------------------------------------------

def test_fiala_is_linear_near_zero_saturates_at_the_friction_limit_and_is_continuous():
    c, peak = 80_000.0, 8_000.0
    assert fiala(0.0, c, peak) == 0.0
    assert fiala(1e-4, c, peak) == pytest.approx(-c * math.tan(1e-4), rel=1e-3)
    sliding = math.atan(3 * peak / c)
    assert fiala(sliding * 1.5, c, peak) == -peak
    assert fiala(-sliding * 1.5, c, peak) == peak
    assert fiala(sliding * (1 - 1e-9), c, peak) == pytest.approx(-peak, rel=1e-6)   # continuous
    forces = [-fiala(sliding * i / 100, c, peak) for i in range(101)]
    assert forces == sorted(forces)                                                # monotone


# --- Steady-state cornering -----------------------------------------------------

@pytest.mark.parametrize(("speed", "steer"), [(10, 0.02), (20, 0.01), (30, 0.005)])
def test_linear_cornering_matches_the_understeer_gradient_equation(speed, steer):
    # Rajamani (2012) eq. 3.15: delta = L / R + K a_y in the linear tire range.
    s = steady_turn(DYN, speed, steer)
    lateral_accel = s.speed * s.yaw_rate
    radius = s.speed / s.yaw_rate
    predicted = DYN.wheelbase / radius + understeer_gradient(DYN) * lateral_accel
    assert predicted == pytest.approx(steer, rel=0.02)


def test_the_car_understeers_and_never_exceeds_the_friction_limit():
    kinematic_radius = DYN.wheelbase / math.tan(0.08)
    s = steady_turn(DYN, 25.0, 0.08)
    assert s.speed / s.yaw_rate > 1.5 * kinematic_radius          # runs wide
    for steer in (0.2, 0.4, 0.6):                                  # far past the grip limit
        car = Vehicle(replace(DYN, max_steer_rate=10.0), VehicleState(speed=25.0))
        peak = 0.0
        for _ in range(1000):
            car.step(VehicleCommand(steer=steer), 0.01)
            peak = max(peak, abs(car.state.lateral_accel))
        assert peak <= DYN.friction * GRAVITY + 1e-9


def test_steady_turn_rear_slip_matches_its_gradient():
    # Linear-range check (a_y ~ 0.5 m/s^2); at higher a_y the Fiala tire softens
    # and needs a few percent more slip than the linear gradient predicts.
    s = steady_turn(DYN, 20.0, 0.005)
    lr = DYN.wheelbase - DYN.cg_to_front
    rear_slip = math.atan2(s.lateral_velocity - lr * s.yaw_rate, s.speed)
    assert -rear_slip == pytest.approx(rear_slip_gradient(DYN) * s.speed * s.yaw_rate, rel=0.03)


def test_low_speed_motion_blends_to_the_kinematic_model():
    kin = Vehicle(CFG.vehicle, VehicleState(speed=1.5))
    dyn = Vehicle(DYN, VehicleState(speed=1.5))
    for _ in range(1000):                                          # 10 s of a tight, slow turn
        kin.step(VehicleCommand(steer=0.4), 0.01)
        dyn.step(VehicleCommand(steer=0.4), 0.01)
    assert math.hypot(kin.state.x - dyn.state.x, kin.state.y - dyn.state.y) < 0.05


def test_the_dynamic_model_is_well_behaved_at_a_standstill_and_pulling_away():
    car = Vehicle(DYN, VehicleState())
    for k in range(600):
        car.step(VehicleCommand(accel=1.5 if k > 100 else 0.0, steer=0.3), 0.01)
        s = car.state
        assert all(math.isfinite(v) for v in (s.x, s.y, s.yaw, s.speed, s.lateral_velocity, s.yaw_rate))
    assert car.state.speed > 5.0


def test_the_kinematic_model_reports_its_yaw_rate():
    car = Vehicle(CFG.vehicle, VehicleState(speed=12.0))
    car.step(VehicleCommand(steer=0.05), 0.01)
    s = car.state
    assert s.yaw_rate == s.speed * math.tan(s.steer) / CFG.vehicle.wheelbase


# --- The stack on the dynamic plant ---------------------------------------------

def test_the_steering_cap_holds_lateral_acceleration_to_the_envelope_on_a_slipping_car():
    cfg = use_dynamic_vehicle(CFG)
    sup = SafetySupervisor(cfg.safety, cfg.vehicle)
    speed = 25.0
    cap = sup._limit_steer(1.0, speed, 1.0)       # rate limit wide open for one long step
    s = steady_turn(cfg.vehicle, speed, cap)
    # The cap uses the linear understeer gradient. Real tires soften near 0.3 g,
    # so the car achieves a little less than the limit: conservative, as a
    # safety envelope should be, and within 10 % of it.
    achieved = abs(s.speed * s.yaw_rate)
    assert 0.9 * cfg.safety.max_lat_accel_cmd <= achieved <= cfg.safety.max_lat_accel_cmd


@pytest.mark.parametrize("name", list(scenarios.ALL))
def test_every_scenario_passes_on_the_dynamic_plant(name):
    result = run(scenarios.ALL[name](), use_dynamic_vehicle(CFG))
    assert result.passed, result.summary()
    assert result.max_lateral_error < 0.2
    if name != "sensor_failure":
        assert "fault" not in result.log["mode"]
