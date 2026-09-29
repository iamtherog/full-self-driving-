"""Other road users. Ground truth that the ego car can only see through sensors."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SpeedChange:
    t: float             # s, when the lead driver starts changing speed
    target: float        # m/s
    accel: float         # m/s^2 magnitude used to get there


@dataclass
class LeadVehicle:
    """A car in the ego lane that drives along the route with a scripted speed plan.

    `appear_at` models a cut-in: the car is invisible (not in our lane) until
    then, and merges in `appear_gap` meters ahead of the ego bumper.
    `leave_at` models it turning off the road.
    """
    s: float = 60.0
    speed: float = 20.0
    length: float = 4.7
    plan: list[SpeedChange] = field(default_factory=list)
    appear_at: float = 0.0
    appear_gap: float | None = None
    leave_at: float = float("inf")
    present: bool = field(init=False, default=False)
    _target: float = field(init=False, default=0.0)
    _rate: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._target, self._rate = self.speed, 0.0
        self.plan = sorted(self.plan, key=lambda c: c.t)

    def step(self, t: float, dt: float, ego_front_s: float) -> None:
        if t >= self.leave_at:
            self.present = False
            return
        if not self.present and t >= self.appear_at:
            self.present = True
            if self.appear_gap is not None:
                self.s = ego_front_s + self.appear_gap + self.length
        while self.plan and self.plan[0].t <= t:
            change = self.plan.pop(0)
            self._target, self._rate = change.target, change.accel
        if self.speed < self._target:
            self.speed = min(self._target, self.speed + self._rate * dt)
        elif self.speed > self._target:
            self.speed = max(self._target, self.speed - self._rate * dt)
        self.s += self.speed * dt

    @property
    def rear_s(self) -> float:
        return self.s - self.length
