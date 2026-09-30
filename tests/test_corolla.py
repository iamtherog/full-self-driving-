"""The simulated car against published 2026 Toyota Corolla sedan specifications.

Published figures (2026 Corolla LE, 2.0 L CVT, from spec aggregators and
instrumented road tests; see README, "Checked against a 2026 Corolla"):
wheelbase 106 in, length 182.5 in, width 70.1 in, curb weight 2,955 lb,
turning diameter 36 ft, 0-60 mph 7.8-8.9 s.
"""

import math
from dataclasses import replace

import pytest

from autodrive.config import VehicleParams
from autodrive.vehicle import Vehicle, VehicleCommand, VehicleState

IN, FT, LB, MPH = 0.0254, 0.3048, 0.45359237, 0.44704
P = VehicleParams()


def test_dimensions_match_published_specs():
    assert P.wheelbase == pytest.approx(106 * IN, rel=0.01)
    assert P.length == pytest.approx(182.5 * IN, rel=0.01)
    assert P.width == pytest.approx(70.1 * IN, rel=0.01)


def test_mass_is_the_published_curb_weight_plus_a_driver():
    assert P.mass == pytest.approx(2955 * LB + 75.0, abs=5.0)


def test_steering_lock_reproduces_the_published_turning_circle():
    # Outer front wheel radius for a bicycle model at full lock, with the tire's
    # outer edge somewhere between 0.78 m and the half body width from center.
    rear_radius = P.wheelbase / math.tan(P.max_steer)
    for half_track in (0.78, P.width / 2):
        diameter = 2 * math.hypot(rear_radius + half_track, P.wheelbase)
        assert diameter == pytest.approx(36 * FT, rel=0.02)


@pytest.mark.parametrize("model", ["kinematic", "dynamic"])
def test_the_simulated_car_is_not_quicker_than_the_real_one(model):
    # A simulator that out-accelerates the car it models flatters every result.
    car = Vehicle(replace(P, model=model), VehicleState())
    t = 0.0
    while car.state.speed < 60 * MPH:
        car.step(VehicleCommand(accel=10.0), 0.01)
        t += 0.01
    fastest_published = 7.8
    assert fastest_published <= t <= 10.0
