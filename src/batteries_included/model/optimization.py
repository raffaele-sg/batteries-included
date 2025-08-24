from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from functools import cached_property
from itertools import pairwise
from typing import Any, Literal, Self

import linopy
import pandas as pd
import xarray as xr

from batteries_included.model.common import Battery, PriceScenarios

fraction = float


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
            upper=self.battery.power,
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
            upper=self.battery.size,
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
            upper=self.battery.size,
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
            upper=self.battery.power,
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

    def _constrain_storage_level_start(
        self,
        soc: tuple[Literal["==", ">=", "<="], fraction],
    ) -> Self:
        sign, _soc = soc
        assert 0.0 <= _soc <= 1.0

        m = self._model
        level_initial = self._var_level_initial
        size = self.battery.size

        m.add_constraints(
            lhs=level_initial,
            sign=sign,
            rhs=_soc * size,
            name="level (initial)",
        )
        return self

    def _constrain_storage_level_end(
        self,
        soc: tuple[Literal["==", ">=", "<="], fraction],
    ) -> Self:
        sign, _soc = soc
        assert 0.0 <= _soc <= 1.0

        m = self._model
        level = self._var_level
        time = self._idx_time
        size = self.battery.size

        m.add_constraints(
            lhs=level.loc[time[-1], :],
            sign=sign,
            rhs=_soc * size,
            name="level (last)",
        )
        return self

    def constrain_storage_level(
        self,
        soc_start: None | tuple[Literal["==", ">=", "<="], fraction] = None,
        soc_end: None | tuple[Literal["==", ">=", "<="], fraction] = None,
    ) -> Self:
        model = self._model
        level = self._var_level
        time = self._idx_time
        scenario = self._idx_scenario

        direction = self._idx_direction

        charge = slice_variable(
            self._var_dispatch,
            dim_name=direction.name,
            dim_value=BidDirection.buy.value,
        ).mul(self.battery.efficiency_charge)

        discharge = slice_variable(
            self._var_dispatch,
            dim_name=direction.name,
            dim_value=BidDirection.sell.value,
        ).div(self.battery.efficiency_discharge)

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

        if soc_start is not None:
            self._constrain_storage_level_start(soc_start)

        if soc_end is not None:
            self._constrain_storage_level_end(soc_end)

        return self

    def _accept_all(self) -> Self:
        bid_accepted = self._var_bid_accepted
        m = self._model
        m.add_constraints(
            lhs=bid_accepted,
            sign="==",
            rhs=1,
            name="accept all",
        )

        return self

    def _constrain_bid_quantity_accepted(self) -> Self:
        accepted = self._var_bid_accepted
        bid_quantity_accepted = self._var_bid_quantity_accepted
        bid_quantity = self._var_bid_quantity

        M = self.battery.power
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
            name="bidqa (a=1 -> gte bidq)",
        )
        m.add_constraints(
            lhs=bid_quantity_accepted - bid_quantity - M * (-accepted + 1),
            sign="<=",
            rhs=0,
            name="bidqa (a=1 -> lte bidq)",
        )

        return self

    def _constrain_bid_price(self) -> Self:
        price = self.price_scenarios.data_array

        m = self._model
        accepted = self._var_bid_accepted

        # bid_quantity_accepted = self._var_bid_quantity_accepted
        # bid_quantity = self._var_bid_quantity
        # bid_price = self._var_bid_price
        # M = price.max()

        # # TOLLERENCE = 0.00

        # bid_price_sell = slice_variable(
        #     bid_price,
        #     dim_name=self._idx_direction.name,
        #     dim_value=BidDirection.sell.value,
        # )
        # bid_price_buy = slice_variable(
        #     bid_price,
        #     dim_name=self._idx_direction.name,
        #     dim_value=BidDirection.buy.value,
        # )

        # accepted_sell = slice_variable(
        #     accepted,
        #     dim_name=self._idx_direction.name,
        #     dim_value=BidDirection.sell.value,
        # )
        # accepted_buy = slice_variable(
        #     accepted,
        #     dim_name=self._idx_direction.name,
        #     dim_value=BidDirection.buy.value,
        # )

        # m.add_constraints(
        #     lhs=bid_price_sell - price + M * accepted_sell,
        #     sign=">=",
        #     rhs=+TOLLERENCE,
        #     name="pb sell (a=0 -> gte p)",
        # )
        # m.add_constraints(
        #     lhs=bid_price_buy - price - M * accepted_buy,
        #     sign="<=",
        #     rhs=-TOLLERENCE,
        #     name="pb buy (a=0 -> lte p)",
        # )

        # m.add_constraints(
        #     lhs=bid_price_sell - price - M * (-accepted_sell + 1),
        #     sign="<=",
        #     rhs=0,
        #     name="pb sell (a=1 -> lte p)",
        # )

        # m.add_constraints(
        #     lhs=bid_price_buy - price + M * (-accepted_buy + 1),
        #     sign=">=",
        #     rhs=0,
        #     name="pb sell (a=1 -> gte p)",
        # )

        direction_name = self._idx_direction.name

        scenario_coord_name = self._idx_scenario.name
        scenario_coord_position = price.get_axis_num(scenario_coord_name)

        sorting_index = price.argsort(scenario_coord_position).assign_coords(
            {scenario_coord_name: range(len(price[scenario_coord_name]))}
        )

        for a, b in pairwise(sorting_index.coords[scenario_coord_name].values):
            idx_lower = sorting_index.sel({scenario_coord_name: a})
            idx_upper = sorting_index.sel({scenario_coord_name: b})

            # If a buy bid is accepted with a higher price, it must also be accepted with a lower price
            m.add_constraints(
                lhs=(
                    accepted.sel({direction_name: BidDirection.buy.value}).isel(
                        {scenario_coord_name: idx_lower}
                    )
                    - accepted.sel({direction_name: BidDirection.buy.value}).isel(
                        {scenario_coord_name: idx_upper}
                    )
                ),
                sign=">=",
                rhs=0,
                name=f"accepted buy order {a}-{b}",
            )

            # If a sell bid is accepted with a lower price, it must also be accepted with a higher price
            m.add_constraints(
                lhs=(
                    accepted.sel({direction_name: BidDirection.sell.value}).isel(
                        {scenario_coord_name: idx_lower}
                    )
                    - accepted.sel({direction_name: BidDirection.sell.value}).isel(
                        {scenario_coord_name: idx_upper}
                    )
                ),
                sign="<=",
                rhs=0,
                name=f"accepted sell order {a}-{b}",
            )

        return self

    def constrain_bidding_strategy(self) -> Self:
        self._constrain_bid_price()
        self._constrain_bid_quantity_accepted()
        return self

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
            name="imbalance",
        )
        return self

    def expected_trade_profits(self) -> linopy.LinearExpression:
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

    def expected_imbalance(self):
        time_scaling = self.price_scenarios.resolution / timedelta(hours=1)
        imbalance = self._var_imbalance

        probabilities_xr = xr.DataArray(
            self.price_scenarios.probabilities,
            coords={self._idx_scenario.name: self._idx_scenario},
            dims=[self._idx_scenario.name],
        )

        return time_scaling * probabilities_xr * imbalance

    def add_objective(self, penalize_imbalance: float) -> Self:
        if penalize_imbalance < abs(self.price_scenarios.data_array).max():
            logging.warning(
                "The imbalance penalty is lower the the highest (abs) price"
            )

        imbalance_cost = penalize_imbalance * self.expected_imbalance().sum()

        self._model.add_objective(
            self.expected_trade_profits() - imbalance_cost,
            sense="max",
        )

        return self

    def solve(self) -> Solution:
        self._model.solve(output_flag=False)

        if self._var_imbalance.solution.max() > 0:
            logging.warning(
                "Probably some imbalances occurred somewhere, consider increasing the penaly"
            )

        return Solution(self)


class ExtractionError(Exception):
    """Indicate a problem in retrieving data from the solved model"""


@dataclass
class Solution:
    _model_builder: ModelBuilder

    def plausible_bidding_prices(
        self,
        margin: None | float,
        remove_quantity_bids_below: None | float,
    ) -> xr.DataArray:
        model_builder = self._model_builder
        bid_accepted = model_builder._var_bid_accepted

        if "solution" not in bid_accepted.data.data_vars:
            raise ExtractionError(
                "A plausible bidding strategy requires the constraint 'constrain_bidding_strategy' "
                "(_var_bid_accepted has no solution)"
            )

        price_bid = xr.DataArray(
            dims=[
                model_builder._idx_time.name,
                model_builder._idx_direction.name,
                "bounds",
            ],
            coords=[
                model_builder._idx_time.values,
                model_builder._idx_direction.values,
                ["lower", "upper"],
            ],
        )

        accepted_prices = model_builder.price_scenarios.data_array.where(
            bid_accepted.solution > 0.5
        )
        rejected_prices = model_builder.price_scenarios.data_array.where(
            bid_accepted.solution < 0.5
        )

        price_bid.sel(
            {
                model_builder._idx_direction.name: BidDirection.sell.value,
                "bounds": "lower",
            }
        ).loc[:] = rejected_prices.max(dim=[model_builder._idx_scenario.name]).sel(
            {model_builder._idx_direction.name: BidDirection.sell.value}
        )
        price_bid.sel(
            {
                model_builder._idx_direction.name: BidDirection.sell.value,
                "bounds": "upper",
            }
        ).loc[:] = accepted_prices.min(dim=[model_builder._idx_scenario.name]).sel(
            {model_builder._idx_direction.name: BidDirection.sell.value}
        )

        price_bid.sel(
            {
                model_builder._idx_direction.name: BidDirection.buy.value,
                "bounds": "upper",
            }
        ).loc[:] = rejected_prices.min(dim=[model_builder._idx_scenario.name]).sel(
            {model_builder._idx_direction.name: BidDirection.buy.value}
        )
        price_bid.sel(
            {
                model_builder._idx_direction.name: BidDirection.buy.value,
                "bounds": "lower",
            }
        ).loc[:] = accepted_prices.max(dim=[model_builder._idx_scenario.name]).sel(
            {model_builder._idx_direction.name: BidDirection.buy.value}
        )

        if margin is not None:
            price_bid.sel({"bounds": "lower"}).loc[:] = price_bid.sel(
                {"bounds": "lower"}
            ).fillna(price_bid.sel({"bounds": "upper"}) - 2 * margin)
            price_bid.sel({"bounds": "upper"}).loc[:] = price_bid.sel(
                {"bounds": "upper"}
            ).fillna(price_bid.sel({"bounds": "lower"}) + 2 * margin)

        price_bid = price_bid.mean(dim=["bounds"])

        if remove_quantity_bids_below is not None:
            price_bid = price_bid.where(
                model_builder._var_bid_quantity.solution >= remove_quantity_bids_below
            )

        return price_bid
