from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

import linopy
import numpy as np
import pandas as pd

from src.model import Battery, EURperkWh, TimeSeries, kWh


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

    m.add_constraints(
        lhs=level.at[time[-1]],
        sign="==",
        rhs=battery.state.soc * size,
        name="level (last)",
    )

    prices = np.array(price.values)

    m.add_objective(sell.dot(prices).sub(buy.dot(prices)), sense="max")
    return m


@dataclass
class Extractor:
    model: linopy.Model
    solve: bool = True

    @staticmethod
    def from_inputs(battery: Battery, price: TimeSeries[EURperkWh]) -> Extractor:
        model = build_dispatch(battery=battery, price=price)
        return Extractor(model=model)

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
