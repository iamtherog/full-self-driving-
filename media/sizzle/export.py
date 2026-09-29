"""Run the scenarios and export the frames the sizzle video needs as JSON.

Every number the video shows comes from these real simulation runs.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autodrive import scenarios, sim  # noqa: E402
from autodrive.config import Config  # noqa: E402

FPS = 30
CFG = Config()
FRONT = CFG.vehicle.wheelbase + CFG.vehicle.front_overhang

# (id, scenario, sim start, sim end, video start, video end)
SHOTS = [
    ("open",     "highway",        4.0, 10.0,  0.0,  3.4),
    ("lane",     "highway",       12.0, 22.0,  3.4,  8.0),
    ("brake",    "hard_brake",    11.2, 17.6,  8.0, 13.0),
    ("cutin",    "cut_in",        10.8, 15.0, 13.0, 16.0),
    ("light",    "city",          13.0, 21.5, 16.0, 18.2),
    ("turn",     "city",          35.5, 42.5, 18.2, 20.4),
    ("blind",    "sensor_failure", 10.6, 16.6, 20.4, 24.4),
]


class RecordingVehicle(sim.Vehicle):
    yaws: list[float] = []

    def step(self, cmd, dt):
        state = super().step(cmd, dt)
        RecordingVehicle.yaws.append(state.yaw)
        return state


def run(name):
    RecordingVehicle.yaws = []
    sim.Vehicle = RecordingVehicle
    scenario = scenarios.ALL[name]()
    result = sim.run(scenario)
    return scenario, result, list(RecordingVehicle.yaws)


def lerp(a, b, u):
    return a + (b - a) * u


def angle_lerp(a, b, u):
    d = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + d * u


def main(out):
    runs, routes, shots = {}, {}, []
    for sid, name, t0, t1, v0, v1 in SHOTS:
        if name not in runs:
            runs[name] = run(name)
            scenario = runs[name][0]
            r = scenario.route
            routes[name] = {
                "s": [round(v, 3) for v in r.s], "x": [round(v, 3) for v in r.x],
                "y": [round(v, 3) for v in r.y], "yaw": [round(v, 5) for v in r.yaw],
                "lights": [l.s for l in r.lights],
            }
        scenario, result, yaws = runs[name]
        log, lead = result.log, scenario.lead
        n = round((v1 - v0) * FPS)
        frames, cam_yaw = [], None
        for i in range(n):
            ts = lerp(t0, t1, i / n)
            k = min(int(ts / 0.01), len(log["t"]) - 2)
            u = (ts - log["t"][k]) / 0.01
            g = lambda key: lerp(log[key][k], log[key][k + 1], u)  # noqa: E731
            yaw = angle_lerp(yaws[k], yaws[k + 1], u)
            cam_yaw = yaw if cam_yaw is None else angle_lerp(cam_yaw, yaw, 0.12)
            gap = log["gap"][k]
            lead_s = None
            if lead is not None and math.isfinite(gap) and gap < 250:
                lead_s = g("s") + FRONT + g("gap") + lead.length / 2
            lights = [l.state(ts) for l in scenario.route.lights]
            ttc = log["ttc"][k]
            frames.append({
                "t": round(ts, 3), "x": round(g("x"), 3), "y": round(g("y"), 3),
                "yaw": round(yaw, 5), "cam": round(cam_yaw, 5), "s": round(g("s"), 2),
                "v": round(g("speed"), 3), "a": round(g("accel"), 3), "steer": round(g("steer"), 4),
                "lat": round(g("lateral"), 4), "mode": log["mode"][k], "aeb": bool(log["aeb"][k]),
                "alert": log["alert"][k], "beh": log["behavior"][k],
                "cap": round(log["speed_cap"][k], 2) if math.isfinite(log["speed_cap"][k]) else None,
                "gap": round(g("gap"), 2) if lead_s is not None else None,
                "lead_s": round(lead_s, 2) if lead_s is not None else None,
                "lead_v": round(log["lead_speed"][k], 2) if lead_s is not None else None,
                "ttc": round(ttc, 2) if math.isfinite(ttc) and ttc < 20 else None,
                "lights": lights,
                "blind": bool(scenario.sensor_dropout
                              and scenario.sensor_dropout[0] <= ts < scenario.sensor_dropout[1]),
            })
        shots.append({"id": sid, "scenario": name, "v0": v0, "v1": v1,
                      "rate": round((t1 - t0) / (v1 - v0), 2), "frames": frames})

    summary = {name: {"passed": r.passed, "max_lat": round(float(r.max_lateral_error), 3),
                      "min_gap": round(r.min_gap, 2) if math.isfinite(r.min_gap) else None,
                      "aeb": r.aeb_activations, "collision": r.collision}
               for name, (_, r, _) in ((n, run(n)) for n in scenarios.ALL)}
    Path(out).write_text(json.dumps({"fps": FPS, "routes": routes, "shots": shots,
                                     "summary": summary}, separators=(",", ":")))
    print(f"wrote {out}: {sum(len(s['frames']) for s in shots)} sim frames")
    for name, s in summary.items():
        print(f"  {name:16s} {s}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data.json")
