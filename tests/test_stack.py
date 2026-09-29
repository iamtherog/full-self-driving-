import math

import pytest

from autodrive import Config, scenarios
from autodrive.control import LateralController, LongitudinalController
from autodrive.perception import LaneEstimate, LeadEstimate, LightEstimate, PerceptionFrame, dead_reckon
from autodrive.planner import Planner
from autodrive.route import Route, Segment, TrafficLight
from autodrive.safety import Mode, SafetySupervisor
from autodrive.sim import Scenario, run
from autodrive.vehicle import DriverInput, Vehicle, VehicleCommand, VehicleState
from autodrive.world import LeadVehicle

CFG = Config()


def frame(t=0.0, lateral=0.0, s=0.0, lead=None, light=None):
    return PerceptionFrame(t, LaneEstimate(s, lateral, 0.0, 0.0), lead, light)


# --- Vehicle model ---------------------------------------------------------

def test_vehicle_drives_straight_and_respects_actuator_limits():
    car = Vehicle(CFG.vehicle, VehicleState(speed=10.0))
    for _ in range(100):
        car.step(VehicleCommand(accel=50.0, steer=0.0), 0.01)
    assert car.state.y == pytest.approx(0.0)
    assert car.state.accel <= CFG.vehicle.max_accel + 1e-9
    assert car.state.x > 10.0


def test_vehicle_never_rolls_backward():
    car = Vehicle(CFG.vehicle, VehicleState(speed=1.0))
    for _ in range(500):
        car.step(VehicleCommand(accel=-8.0), 0.01)
    assert car.state.speed == 0.0


# --- Route geometry --------------------------------------------------------

def test_quarter_circle_ends_at_expected_point():
    r = 50.0
    route = Route([Segment(math.pi / 2 * r, 1 / r, 10.0)])
    x, y, yaw = route.pose_at(route.length)
    assert (x, y, yaw) == pytest.approx((r, r, math.pi / 2), abs=1e-6)


def test_projection_signs_lateral_offset_left_positive():
    route = Route([Segment(100, 0.0, 10.0)])
    fp = route.project(20.0, 0.7, 0.1)
    assert fp.s == pytest.approx(20.0, abs=0.01)
    assert fp.lateral == pytest.approx(0.7)
    assert fp.heading_error == pytest.approx(0.1)


def test_traffic_light_cycle():
    light = TrafficLight(s=0, green=10, yellow=3, red=5)
    assert [light.state(t) for t in (0, 11, 14, 18.5)] == ["green", "yellow", "red", "green"]


# --- Perception ------------------------------------------------------------

def test_dead_reckoning_tracks_straight_road():
    lane = LaneEstimate(0.0, 0.2, 0.0, 0.0)
    for _ in range(100):
        lane = dead_reckon(lane, 10.0, 0.0, 0.0, 0.01)
    assert lane.s == pytest.approx(10.0)
    assert lane.lateral == pytest.approx(0.2)


# --- Planner ---------------------------------------------------------------

def test_planner_slows_for_curve_ahead():
    route = Route([Segment(200, 0.0, 30.0), Segment(200, 1 / 50, 30.0)])
    cap, reason = Planner(route, CFG.planner).speed_cap(190.0, 25.0)
    assert reason == "curve"
    assert cap < 12.0  # sqrt(2.0 m/s^2 * 50 m) = 10 m/s plus a little decel margin


def test_planner_brakes_for_slow_lead():
    route = Route([Segment(500, 0.0, 30.0)])
    lead = LeadEstimate(gap=20.0, rel_speed=-10.0, lead_speed=15.0)
    plan = Planner(route, CFG.planner).plan(frame(lead=lead), 25.0)
    assert plan.behavior == "follow"
    assert plan.accel < -1.0


def test_planner_stops_for_red_but_commits_to_close_yellow():
    route = Route([Segment(500, 0.0, 15.0)])
    planner = Planner(route, CFG.planner)
    red = planner.plan(frame(light=LightEstimate(60.0, "red")), 12.0)
    assert red.behavior == "stop_for_light" and red.accel < 0
    close_yellow = planner.plan(frame(light=LightEstimate(8.0, "yellow")), 13.0)
    assert close_yellow.behavior != "stop_for_light"


# --- Controllers -----------------------------------------------------------

def test_lateral_controller_steers_back_toward_center():
    lat = LateralController(CFG.controller, CFG.vehicle)
    assert lat.update(LaneEstimate(0, 0.5, 0, 0), 0.0, 20.0) < 0   # left of center -> steer right
    assert lat.update(LaneEstimate(0, -0.5, 0, 0), 0.0, 20.0) > 0


def test_longitudinal_controller_is_jerk_limited():
    lon = LongitudinalController(CFG.controller)
    cmd = lon.update(-3.0, 0.0, 0.01)
    assert cmd == pytest.approx(-CFG.controller.max_jerk * 0.01)


# --- Safety supervisor -----------------------------------------------------

def engaged_supervisor():
    sup = SafetySupervisor(CFG.safety, CFG.vehicle)
    sup.update(0.0, frame(), DriverInput(engage_button=True), VehicleCommand(), VehicleState(), 0.01)
    assert sup.mode == Mode.ENGAGED
    return sup


def test_command_envelope_is_enforced():
    sup = engaged_supervisor()
    ego = VehicleState(speed=30.0)
    out = sup.update(0.01, frame(0.01), DriverInput(), VehicleCommand(accel=-9.0, steer=0.5), ego, 0.01)
    assert out.command.accel == pytest.approx(-CFG.safety.max_jerk_cmd * 0.01)
    assert abs(out.command.steer) <= CFG.safety.max_steer_rate_cmd * 0.01 + 1e-9
    for k in range(2, 200):
        out = sup.update(k * 0.01, frame(k * 0.01), DriverInput(),
                         VehicleCommand(accel=-9.0, steer=0.5), ego, 0.01)
    assert out.command.accel == -CFG.safety.max_decel_cmd


def test_supervisor_rate_limits_accel_steps():
    sup = engaged_supervisor()
    out = sup.update(0.01, frame(0.01), DriverInput(), VehicleCommand(accel=2.0),
                     VehicleState(speed=20), 0.01)
    assert out.command.accel == pytest.approx(CFG.safety.max_jerk_cmd * 0.01)


def test_brake_pedal_cancels_and_steering_overrides():
    sup = engaged_supervisor()
    steer = DriverInput(steering_torque=3.0)
    out = sup.update(0.01, frame(0.01), steer, VehicleCommand(), VehicleState(), 0.01)
    assert out.mode == Mode.LAT_OVERRIDE and not out.controls_lateral and out.controls_longitudinal
    brake = DriverInput(brake_pressed=True)
    out = sup.update(0.02, frame(0.02), brake, VehicleCommand(), VehicleState(), 0.01)
    assert out.mode == Mode.OFF and not out.controls_longitudinal


def test_cannot_engage_without_perception():
    sup = SafetySupervisor(CFG.safety, CFG.vehicle)
    sup.update(0.0, None, DriverInput(engage_button=True), VehicleCommand(), VehicleState(), 0.01)
    assert sup.mode == Mode.OFF


def test_stale_perception_faults_and_slows_down():
    sup = engaged_supervisor()
    out = sup.update(1.0, frame(0.0), DriverInput(), VehicleCommand(accel=1.0), VehicleState(speed=20), 0.01)
    assert out.mode == Mode.FAULT and out.alert.startswith("TAKE CONTROL")
    assert out.command.accel < 0.0   # ramping down, jerk-limited
    for k in range(1, 100):
        out = sup.update(1.0 + k * 0.01, frame(0.0), DriverInput(), VehicleCommand(accel=1.0),
                         VehicleState(speed=20), 0.01)
    assert out.command.accel == -CFG.safety.fault_decel


def test_aeb_fires_even_when_disengaged():
    sup = SafetySupervisor(CFG.safety, CFG.vehicle)
    lead = LeadEstimate(gap=10.0, rel_speed=-10.0, lead_speed=0.0)   # TTC = 1 s
    out = sup.update(0.0, frame(lead=lead), DriverInput(), VehicleCommand(), VehicleState(speed=10), 0.01)
    assert out.mode == Mode.OFF and out.aeb_active
    assert out.command.accel == -CFG.safety.aeb_decel


# --- Closed loop -----------------------------------------------------------

@pytest.mark.parametrize("name", list(scenarios.ALL))
def test_scenario_passes(name):
    result = run(scenarios.ALL[name]())
    assert result.passed, result.summary()
    assert result.max_lateral_error < 0.5


def test_hard_brake_needs_and_triggers_aeb():
    assert run(scenarios.hard_brake()).aeb_activations >= 1


def test_normal_driving_is_comfortable():
    result = run(scenarios.highway())
    assert result.aeb_activations == 0
    assert result.max_jerk < 3.0
    assert result.min_gap > 10.0


@pytest.mark.parametrize("name", list(scenarios.ALL))
def test_system_commands_respect_jerk_limit(name):
    # The actuator lag smooths commands, so actual jerk never exceeds the command limit.
    assert run(scenarios.ALL[name]()).max_jerk <= CFG.safety.max_jerk_cmd + 1e-6


def test_stops_behind_stopped_car():
    road = Route([Segment(600, 0.0, 25.0)])
    lead = LeadVehicle(s=150, speed=0.0)
    result = run(Scenario("stopped_car", "", road, duration=40, initial_speed=20, lead=lead))
    assert result.passed and result.aeb_activations == 0
    assert 2.0 < result.min_gap < 8.0


def test_sensor_failure_hands_back_control():
    result = run(scenarios.sensor_failure())
    assert "fault" in result.log["mode"] and result.final_mode == "off"


def test_results_are_deterministic():
    a, b = run(scenarios.cut_in()), run(scenarios.cut_in())
    assert a.log["speed"] == b.log["speed"]
