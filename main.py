from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import NamedTuple, Sequence, Type

import linopy
import numpy as np
import pandas as pd
import xarray

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


class Variables(Enum):
    buy = "buy"
    sell = "sell"
    level = "level"


def build_dispatch(battery: Battery, price: TimeSeries[EURperkWh]) -> linopy.Model:
    m = linopy.Model()
    time = pd.Index(np.arange(len(price.values)), name="time")

    size: kWh = battery.parameters.power * (
        battery.parameters.duration / timedelta(hours=1)
    )

    hours = price.resolution / timedelta(hours=1)

    buy: linopy.Variable = m.add_variables(
        name=Variables.buy.value,
        lower=0.0,
        upper=battery.parameters.power,
        coords=[time],
    )
    sell: linopy.Variable = m.add_variables(
        name=Variables.sell.value,
        lower=0.0,
        upper=battery.parameters.power,
        coords=[time],
    )

    level: linopy.Variable = m.add_variables(
        name=Variables.level.value,
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

    prices = np.array(price.values)

    m.add_objective(sell.dot(prices).sub(buy.dot(prices)), sense="max")
    return m


@dataclass
class Extractor:
    model: linopy.Model
    solve: bool = True

    def __post_init__(self):
        if self.solve:
            self.model.solve(output_flag=False)

        assert self.model.status == "ok"

    def profit(self) -> float:
        x = self.model.objective.value
        assert x is not None
        return float(x)

    def to_numpy(self, variable: Variables) -> np.typing.NDArray[np.float64]:
        x = self.model.solution.get(variable.value)
        assert x is not None
        return np.array(x.values, dtype=np.float64)


def main():
    model = build_dispatch(battery=Battery.example(), price=TimeSeries.example())
    print(model.status)
    model.solve(output_flag=False)
    print(model.status)

    x: xarray.Dataset = model.solution
    print(type(x))
    print(x.get("level"))


if __name__ == "__main__":
    main()
