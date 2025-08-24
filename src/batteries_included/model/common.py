from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from math import pi, sin
from typing import Any, Hashable, Mapping, NamedTuple, Protocol, Sequence, Type

import numpy as np
import pandas as pd
import xarray as xr

MW = float
MWh = float
EURperMWh = float


# class StateOfCharge(NamedTuple):
#     soc: float  # State of charge, [0.0, 1.0]

#     @classmethod
#     def example(cls: Type[StateOfCharge]):
#         return cls(soc=0.5)


# class Battery(NamedTuple):
#     duration: timedelta  # The time it takes to fully discharge the battery
#     power: MW  # represents both, charging and discharing, in kW
#     efficiency: float  # Share of energy losses

#     def size(self) -> MWh:
#         return self.power * (self.duration / timedelta(hours=1))

#     @classmethod
#     def example(cls: Type[Battery]):
#         return cls(duration=timedelta(hours=2), efficiency=0.5**2, power=2.0)


class Battery(NamedTuple):
    duration: timedelta  # The time it takes to fully discharge the battery
    power: MW  # represents both, charging and discharing, in kW
    efficiency: float  # Share of energy losses (round-trip)

    @property
    def size(self) -> MWh:
        return self.power * (self.duration / timedelta(hours=1))

    @property
    def efficiency_charge(self):
        return self.efficiency**0.5

    @property
    def efficiency_discharge(self):
        return self.efficiency**0.5

    @classmethod
    def example(cls: Type[Battery]):
        return cls(
            duration=timedelta(hours=2),
            power=2.0,
            efficiency=0.9,
        )


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


class InconsistentPriceScenarios(Exception):
    "Excation raised becuase price sceanrios are inconsistent"


class Scenario[T](NamedTuple):
    probability: float
    value: T


def time_index(obj: PriceScenarios | TimeSeries):
    start = obj.start
    resolution = obj.resolution
    values = obj.values

    if start is None:
        return [i for i, _ in enumerate(values)]

    return [start + i * resolution for i, _ in enumerate(values)]


@dataclass
class PriceScenarios[T]:
    scenarios: (
        Mapping[Hashable, Scenario[TimeSeries[T]]] | dict[str, Scenario[TimeSeries[T]]]
    )

    @cached_property
    def start(self) -> None | datetime:
        (s,) = {s.value.start for s in self.scenarios.values()}
        return s

    @cached_property
    def resolution(self) -> timedelta:
        (s,) = {s.value.resolution for s in self.scenarios.values()}
        return s

    def time_index(self) -> list[int] | list[datetime]:
        start = self.start
        resolution = self.resolution
        values = self.values

        if start is None:
            return [i for i, _ in enumerate(values)]

        return [start + i * resolution for i, _ in enumerate(values)]

    @cached_property
    def data_array(self) -> xr.DataArray:
        name = "time"
        dim_time, _ = self.values.shape

        time = (
            pd.date_range(
                start=self.start,
                freq=self.resolution,
                periods=dim_time,
                name=name,
            )
            if isinstance(self.start, datetime)
            else pd.Index(np.arange(dim_time), name=name)
        )

        return xr.DataArray(
            [s.value.values for s in self.scenarios.values()],
            dims=["scenario", "time"],
            coords=[
                list(self.scenarios.keys()),
                time,
            ],
        )

    def __post_init__(self):
        absolute_tolerance = 1e-08
        if (
            not abs(sum(s.probability for s in self.scenarios.values()) - 1)
            <= absolute_tolerance
        ):
            raise InconsistentPriceScenarios("Probability does not sum up to 1.0")

        if not len({s.value.start for s in self.scenarios.values()}) == 1:
            raise InconsistentPriceScenarios(
                "TimeSeries have different start attributes"
            )

        if not len({s.value.resolution for s in self.scenarios.values()}) == 1:
            raise InconsistentPriceScenarios(
                "TimeSeries have different resolution attributes"
            )

        if not len({len(s.value.values) for s in self.scenarios.values()}) == 1:
            raise InconsistentPriceScenarios("TimeSeries have different lengths")

    @cached_property
    def probabilities(self):
        return np.array(
            tuple(s.probability for s in self.scenarios.values()), dtype=float
        )

    @cached_property
    def values(self):
        return np.array(
            tuple(s.value.values for s in self.scenarios.values()), dtype=float
        ).T

    def to_array(self) -> xr.DataArray:
        return xr.DataArray([i for i in self.scenarios])
