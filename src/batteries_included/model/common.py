from __future__ import annotations

from datetime import datetime, timedelta
from math import pi, sin
from typing import NamedTuple, Sequence, Type

kW = float
kWh = float
EURperkWh = float


class State(NamedTuple):
    soc: float  # State of charge, [0.0, 1.0]

    @classmethod
    def example(cls: Type[State]):
        return cls(soc=0.5)


class Parameters(NamedTuple):
    duration: timedelta  # The time it takes to fully discharge the battery
    power: kW  # represents both, charging and discharing, in kW
    efficiency: float  # Share of energy losses

    def size(self) -> kWh:
        return self.power * (self.duration / timedelta(hours=1))

    @classmethod
    def example(cls: Type[Parameters]):
        return cls(duration=timedelta(hours=2), efficiency=0.5**2, power=2.0)


class Battery(NamedTuple):
    state: State
    parameters: Parameters

    @classmethod
    def example(cls: Type[Battery]):
        return cls(state=State.example(), parameters=Parameters.example())


class TimeSeries[T](NamedTuple):
    start: datetime | None
    resolution: timedelta
    values: Sequence[T]

    @classmethod
    def example(cls: Type[TimeSeries[float]]):
        return cls(
            start=None,
            resolution=timedelta(minutes=30),
            values=[sin(pi * 2 * i / 48) for i in range(48)],
        )
