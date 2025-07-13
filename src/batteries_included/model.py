from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple, Sequence, Type

kW = float
kWh = float
EURperkWh = float


class BatteryState(NamedTuple):
    soc: float  # State of charge, [0.0, 1.0]

    @classmethod
    def example(cls: Type[BatteryState]):
        return cls(soc=0.5)


class BatteryParameters(NamedTuple):
    duration: timedelta  # The time it takes to fully discharge the battery
    power: kW  # represents both, charging and discharing, in kW
    efficiency: float  # Share of energy losses

    @classmethod
    def example(cls: Type[BatteryParameters]):
        return cls(duration=timedelta(hours=2), efficiency=0.5**2, power=2.0)


class Battery(NamedTuple):
    state: BatteryState
    parameters: BatteryParameters

    @classmethod
    def example(cls: Type[Battery]):
        return cls(state=BatteryState.example(), parameters=BatteryParameters.example())


class TimeSeries[T](NamedTuple):
    start: datetime | None
    resolution: timedelta
    values: Sequence[T]

    @classmethod
    def example(cls: Type[TimeSeries[float]]):
        return cls(
            start=None,
            resolution=timedelta(minutes=30),
            values=[float(i) for i in range(-10, 10)],
        )
