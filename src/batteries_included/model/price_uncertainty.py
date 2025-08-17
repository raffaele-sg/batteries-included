from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import cached_property
from typing import Any, Hashable, NamedTuple, Self

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from batteries_included.model.common import Battery, TimeSeries


class Scenario[T](NamedTuple):
    probability: float
    value: T


class InconsistentPriceScenarios(Exception):
    "Excation raised becuase price sceanrios are inconsistent"


fraction = float


@dataclass
class PriceScenarios[T]:
    scenarios: dict[Hashable, Scenario[TimeSeries[T]]]

    @cached_property
    def start(self) -> None | datetime:
        (s,) = {s.value.start for s in self.scenarios.values()}
        return s

    @cached_property
    def resolution(self) -> timedelta:
        (s,) = {s.value.resolution for s in self.scenarios.values()}
        return s

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


class BidAttribute(Enum):
    price = "price"
    quantity = "quantity"


class BidDirection(Enum):
    buy = "buy"
    sell = "sell"


class Position(Enum):
    long = "long"
    short = "short"


def slice_variable(
    variable: linopy.Variable, dim_name: Any, dim_value: Any
) -> linopy.Variable:
    v = variable.sel({dim_name: dim_value}, drop=True)
    assert isinstance(v, linopy.Variable)

    return v


def slice_expression(
    variable: linopy.LinearExpression, dim_name: Any, dim_value: Any
) -> linopy.LinearExpression:
    v = variable.sel({dim_name: dim_value}, drop=True)
    assert isinstance(v, linopy.LinearExpression)

    return v


@dataclass
class ModelBuilder:
    battery: Battery
    price_scenarios: PriceScenarios[float]

    _model: linopy.Model = field(default_factory=linopy.Model)

    BIG_M: float = 1_000.0

    def reset_model(self) -> ModelBuilder:
        return ModelBuilder(
            battery=self.battery,
            price_scenarios=self.price_scenarios,
        )

    @cached_property
    def _idx_time(self):
        """Time dimension"""
        return self.price_scenarios.data_array.get_index("time")

    @cached_property
    def _idx_scenario(self):
        """Scenario dimension"""
        return self.price_scenarios.data_array.get_index("scenario")

    @cached_property
    def _idx_bid_attribute(self):
        """Bid attribute dimension representing a price and a quantity"""
        return pd.Index([b.value for b in BidAttribute], name="bid_attribute")

    @cached_property
    def _idx_direction(self):
        """Direction dimension representing the buy or sell direction"""
        return pd.Index([b.value for b in BidDirection], name="direction")

    @cached_property
    def _idx_position(self):
        """Direction dimension representing the buy or sell direction"""
        return pd.Index([b.value for b in Position], name="position")

    @cached_property
    def _var_dispatch(self) -> linopy.Variable:
        """
        Acutal energy sold by time and scenario

        Dims:
        - time
        - direction
        - scenario
        """
        return self._model.add_variables(
            name="dispatch",
            lower=0.0,
            upper=self.battery.parameters.power,
            coords=[self._idx_time, self._idx_direction, self._idx_scenario],
        )

    @cached_property
    def _var_level(self) -> linopy.Variable:
        """
        Actual storage level by time and scenario

        Dims:
        - time
        - scenario
        """
        return self._model.add_variables(
            name="level",
            lower=0.0,
            upper=self.battery.parameters.size(),
            coords=[self._idx_time, self._idx_scenario],
        )

    @cached_property
    def _var_level_initial(self):
        """
        Actual initial storage level (Scalar)
        """
        return self._model.add_variables(
            name="level (initial)",
            lower=0.0,
            upper=self.battery.parameters.size(),
            coords=None,
        )

    @cached_property
    def _var_bid_quantity(self) -> linopy.Variable:
        """
        Bid quantity placed by time and direction

        Dims:
        - time
        - direction
        """
        return self._model.add_variables(
            name="bid quantity",
            lower=0.0,
            upper=self.battery.parameters.power,
            coords=[self._idx_time, self._idx_direction],
        )

    @cached_property
    def _var_bid_quantity_accepted(self) -> linopy.Variable:
        """
        Bid quantity accepted by time, direction, and scenario

        Dims:
        - time
        - direction
        - scenaro
        """
        return self._model.add_variables(
            name="bid quantity accepted",
            coords=[self._idx_time, self._idx_direction, self._idx_scenario],
        )

    @cached_property
    def _var_bid_price(self) -> linopy.Variable:
        """
        Bid price placed by time and direction

        Dims:
        - time
        - direction
        """
        return self._model.add_variables(
            name="bid price",
            coords=[self._idx_time, self._idx_direction],
        )

    @cached_property
    def _var_imbalance(self) -> linopy.Variable:
        """
        Imbalance places by time and bid attribute and direction.
        A positive imbalance in the buy direction means that
        more energy is dispatched in the buy direction than the energy bought with the bid.

        Dims:
        - time
        - direction
        - scenario
        - position
        """
        return self._model.add_variables(
            name="imbalance",
            lower=0.0,
            coords=[
                self._idx_time,
                self._idx_direction,
                self._idx_scenario,
                self._idx_position,
            ],
        )

    @cached_property
    def _var_bid_accepted(self) -> linopy.Variable:
        """Whether a bid is accepted or not

        Dims:
        - time
        - scenario
        - direction
        """
        return self._model.add_variables(
            name="bid accepted",
            binary=True,
            coords=[self._idx_time, self._idx_scenario, self._idx_direction],
        )

    def constrain_storage_level(self) -> Self:
        model = self._model
        level = self._var_level
        time = self._idx_time
        scenario = self._idx_scenario

        direction = self._idx_direction

        charge = slice_variable(
            self._var_dispatch,
            dim_name=direction.name,
            dim_value=BidDirection.buy.value,
        ).mul(self.battery.parameters.efficiency**0.5)

        discharge = slice_variable(
            self._var_dispatch,
            dim_name=direction.name,
            dim_value=BidDirection.sell.value,
        ).div(self.battery.parameters.efficiency**0.5)

        hours = self.price_scenarios.resolution / timedelta(hours=1)
        # level_initial = self.battery.state.soc * self.battery.parameters.size()

        # shifted_level: linopy.Variable = level.shift({time.name: 1})
        # print(self._var_level_initial)
        # print(shifted_level.fillna(self._var_level_initial))

        level_shifted = level.shift({time.name: 1}).fillna(self._var_level_initial)
        # print(level_shifted)

        model.add_constraints(
            # lhs=(
            #     level.sub(level.shift({time.name: 1}, fill_value=level_initial))
            #     .sub((charge.sub(discharge)).mul(hours))
            # ),
            lhs=level.sub(level_shifted).sub(charge.sub(discharge).mul(hours)),
            sign="==",
            rhs=0.0,
            coords=[time, scenario],
            name="level",
        )
        return self

    def constrain_storage_level_start(self, soc: fraction) -> Self:
        assert 0.0 <= soc <= 1.0
        m = self._model
        level_initial = self._var_level_initial
        size = self.battery.parameters.size()

        m.add_constraints(
            lhs=level_initial,
            sign="==",
            rhs=soc * size,
            name="level (initial)",
        )
        return self

    def constrain_storage_level_end(self, soc: fraction) -> Self:
        assert 0.0 <= soc <= 1.0
        m = self._model
        level = self._var_level
        time = self._idx_time
        size = self.battery.parameters.size()

        m.add_constraints(
            lhs=level.loc[time[-1], :],
            sign="==",
            rhs=soc * size,
            name="level (last)",
        )
        return self

    def accept_all(self) -> Self:
        bid_accepted = self._var_bid_accepted
        m = self._model
        m.add_constraints(
            lhs=bid_accepted,
            sign="==",
            rhs=1,
            name="accept all",
        )

        return self

    def constraint_bid_quantity_accepted(self) -> Self:
        accepted = self._var_bid_accepted
        bid_quantity_accepted = self._var_bid_quantity_accepted
        bid_quantity = self._var_bid_quantity
        M = self.BIG_M
        m = self._model

        m.add_constraints(
            lhs=bid_quantity_accepted + M * accepted,
            sign=">=",
            rhs=0,
            name="bidqa (a=0 -> gte 0)",
        )
        m.add_constraints(
            lhs=bid_quantity_accepted - M * accepted,
            sign="<=",
            rhs=0,
            name="bidqa (a=0 -> lte 0)",
        )

        m.add_constraints(
            lhs=bid_quantity_accepted - bid_quantity + M * (-accepted + 1),
            sign=">=",
            rhs=0,
            name="bidqa (a=0 -> gte bidq)",
        )

        m.add_constraints(
            lhs=bid_quantity_accepted - bid_quantity - M * (-accepted + 1),
            sign="<=",
            rhs=0,
            name="bidqa (a=0 -> lte bidq)",
        )

        return self

    def contrain_bids(self) -> Self:
        raise NotImplementedError("contrain_bids")

    def trade_profits(self) -> linopy.LinearExpression:
        time_scaling = self.price_scenarios.resolution / timedelta(hours=1)
        probabilities = self.price_scenarios.probabilities
        prices = self.price_scenarios.values

        bid_quantity_accepted = self._var_bid_quantity_accepted

        sell = slice_variable(
            bid_quantity_accepted,
            dim_name=self._idx_direction.name,
            dim_value=BidDirection.sell.value,
        )
        buy = slice_variable(
            bid_quantity_accepted,
            dim_name=self._idx_direction.name,
            dim_value=BidDirection.buy.value,
        )

        return time_scaling * (
            sell.dot(probabilities * prices).sub(buy.dot(probabilities * prices))
        )

    # def bid_price_penalty(self, penalty: float) -> linopy.LinearExpression:
    #     time_scaling = self.price_scenarios.resolution / timedelta(hours=1)
    #     return time_scaling * penalty * self._var_bid_price

    # def bid_quantity_penalty(self, penalty: float) -> linopy.LinearExpression:
    #     time_scaling = self.price_scenarios.resolution / timedelta(hours=1)
    #     return time_scaling * penalty * self._var_bid_quantity

    # def bid_accepted_penalty(self, penalty: float) -> linopy.LinearExpression:
    #     time_scaling = self.price_scenarios.resolution / timedelta(hours=1)
    #     return time_scaling * penalty * self._var_bid_accepted

    def constrain_imbalance(self) -> Self:
        imbalance = self._var_imbalance
        m = self._model
        dispatch = self._var_dispatch
        bid_quantity_accepted = self._var_bid_quantity_accepted

        m.add_constraints(
            lhs=(
                imbalance.sel({self._idx_position.name: Position.long.value})
                - imbalance.sel({self._idx_position.name: Position.short.value})
            )
            - (dispatch - bid_quantity_accepted),
            sign="==",
            rhs=0,
            name="imbalance (long)",
        )
        return self

    def imbalance_penalty(self, penalty: float):
        time_scaling = self.price_scenarios.resolution / timedelta(hours=1)
        imbalance = self._var_imbalance

        probabilities_xr = xr.DataArray(
            self.price_scenarios.probabilities,
            coords={self._idx_scenario.name: self._idx_scenario},
            dims=[self._idx_scenario.name],
        )
        return time_scaling * (penalty) * (probabilities_xr * imbalance)

    def add_objective(self, penalty: float = 1000) -> Self:
        self._model.add_objective(
            self.trade_profits() - self.imbalance_penalty(penalty=penalty),
            # - self.bid_accepted_penalty(10)
            # - self.bid_quantity_penalty(10)
            # - self.bid_price_penalty(10)
            sense="max",
        )

        return self

    def solve(self) -> Solution:
        self._model.solve(output_flag=True)
        return Solution(self)


@dataclass
class Solution:
    _model_builder: ModelBuilder
