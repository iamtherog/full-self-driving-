"""Test drives. Each one exercises a specific capability or failure mode."""

from .route import Route, Segment, TrafficLight
from .sim import Scenario
from .vehicle import DriverInput
from .world import LeadVehicle, SpeedChange

MPH = 0.44704


def highway() -> Scenario:
    road = Route([
        Segment(300, 0.0, 65 * MPH),
        Segment(350, 1 / 600, 65 * MPH),
        Segment(250, 0.0, 65 * MPH),
        Segment(300, -1 / 500, 65 * MPH),
        Segment(400, 0.0, 65 * MPH),
        Segment(200, 1 / 250, 45 * MPH),   # tight exit ramp with a lower limit
        Segment(150, 0.0, 45 * MPH),
    ])
    lead = LeadVehicle(s=90, speed=26, plan=[
        SpeedChange(t=20, target=15, accel=1.5),   # traffic slows ahead
        SpeedChange(t=35, target=27, accel=1.0),
    ])
    return Scenario("highway", "Curvy highway at 65 mph with slowing traffic and an exit ramp.",
                    road, duration=100, initial_speed=25, lead=lead)


def city() -> Scenario:
    road = Route(
        [
            Segment(220, 0.0, 30 * MPH),
            Segment(39.27, 1 / 25, 15 * MPH),   # 90-degree left turn
            Segment(260, 0.0, 30 * MPH),
            Segment(39.27, -1 / 25, 15 * MPH),  # 90-degree right turn
            Segment(200, 0.0, 30 * MPH),
        ],
        lights=[
            TrafficLight(s=180, green=10, yellow=3, red=20, offset=0),   # red on arrival
            # Turns yellow when we're too close to stop comfortably: go through.
            TrafficLight(s=480, green=15, yellow=3, red=15, offset=18.3),
        ],
    )
    lead = LeadVehicle(s=30, speed=8, leave_at=9, plan=[SpeedChange(t=3, target=11, accel=1.0)])
    return Scenario("city", "Urban route with 90-degree turns, traffic lights and a lead car.",
                    road, duration=120, initial_speed=0, lead=lead)


def hard_brake() -> Scenario:
    road = Route([Segment(1200, 0.0, 70 * MPH)])
    lead = LeadVehicle(s=45, speed=29, plan=[SpeedChange(t=10, target=0, accel=8.0)])
    return Scenario("hard_brake", "Lead car panic-stops at 0.8 g; emergency braking must engage.",
                    road, duration=30, initial_speed=29, lead=lead)


def cut_in() -> Scenario:
    road = Route([Segment(1500, 0.0, 60 * MPH)])
    lead = LeadVehicle(speed=17, appear_at=12, appear_gap=14)
    return Scenario("cut_in", "A slower car cuts in 14 m ahead at a 10 m/s speed difference.",
                    road, duration=40, initial_speed=27, lead=lead)


def driver_override() -> Scenario:
    road = Route([Segment(400, 0.0, 55 * MPH), Segment(300, 1 / 300, 55 * MPH),
                  Segment(400, 0.0, 55 * MPH)])

    def driver(t: float) -> DriverInput:
        return DriverInput(
            engage_button=t < 0.5 or 30 <= t < 30.5,     # engage, later re-engage
            steering_torque=2.5 if 10 <= t < 13 else 0.0,  # driver nudges the wheel
            brake_pressed=22 <= t < 25,                  # driver taps the brake
        )

    return Scenario("driver_override", "Driver steers, then brakes (system cancels), then re-engages.",
                    road, duration=45, initial_speed=22, driver=driver)


def sensor_failure() -> Scenario:
    road = Route([Segment(300, 0.0, 55 * MPH), Segment(400, 1 / 800, 55 * MPH),
                  Segment(300, 0.0, 55 * MPH)])

    def driver(t: float) -> DriverInput:
        # Sensors drop at 12 s; the alert fires 0.25 s later and the driver
        # needs ~2 s to react and brake.
        return DriverInput(engage_button=t < 0.5, brake_pressed=t >= 14.5)

    return Scenario("sensor_failure", "Camera and radar go silent; system must alert and slow down.",
                    road, duration=40, initial_speed=24, driver=driver,
                    sensor_dropout=(12.0, 40.0))


ALL = {f.__name__: f for f in (highway, city, hard_brake, cut_in, driver_override, sensor_failure)}
