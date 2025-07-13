from __future__ import annotations
import pandas as pd
from typing import NamedTuple, Sequence, Type

import linopy
import numpy as np
from datetime import datetime, timedelta


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


def build_dispatch(battery: Battery, price: TimeSeries[EURperkWh]) -> linopy.Model:
    m = linopy.Model()
    time = pd.Index(np.arange(len(price.values)), name="time")

    size: kWh = battery.parameters.power * (
        battery.parameters.duration / timedelta(hours=1)
    )

    hours = price.resolution / timedelta(hours=1)

    buy: linopy.Variable = m.add_variables(
        name="buy",
        lower=0.0,
        upper=battery.parameters.power,
        coords=[time],
    )
    sell: linopy.Variable = m.add_variables(
        name="sell",
        lower=0.0,
        upper=battery.parameters.power,
        coords=[time],
    )

    level: linopy.Variable = m.add_variables(
        name="level",
        lower=0.0,
        upper=size,
        coords=[time],
    )

    charge = buy.mul(battery.parameters.efficiency**0.5)
    discharge = sell.div(battery.parameters.efficiency**0.5)

    m.add_constraints(
        lhs=(
            level.sub(
                level.shift({time.name: 1}, fill_value=battery.state.soc * size)
            ).sub(charge.sub(discharge).mul(hours))
        ),
        sign="==",
        rhs=0.0,
        coords=[time],
        name="level",
    )

    m.add_objective(sell.dot(price.values) - buy.dot(price.values), sense="max")  # type: ignore
    return m


def main():
    model = build_dispatch(battery=Battery.example(), price=TimeSeries.example())
    print(model.status)


if __name__ == "__main__":
    main()
