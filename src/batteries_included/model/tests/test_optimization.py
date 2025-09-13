from datetime import datetime, timedelta
from math import pi, sin

import linopy
import pytest

from batteries_included.model.common import (
    Battery,
    PriceScenarios,
    Scenario,
    TimeSeries,
)
from batteries_included.model.optimization import (
    ModelBuilder,
    Position,
    Solution,
)


def create_battery():
    return Battery(
        duration=timedelta(hours=2),
        power=2.0,
        efficiency=0.9,
        variable_cost=1.0,
    )


def create_price_scenarios():
    return TimeSeries(
        start=None,
        resolution=timedelta(minutes=15),
        values=[(sin(pi * 2 * i / 24), 20.0) for i in range(96)],
    )


def create_price_scenarios_with_probaility(start: None | datetime):
    n = 24
    price_scenario_1 = Scenario(
        probability=0.1,
        value=TimeSeries(
            start=start,
            resolution=timedelta(minutes=15),
            values=[sin(pi * 2 * i / n) for i in range(n)],
        ),
    )
    price_scenario_2 = Scenario(
        probability=0.6,
        value=TimeSeries(
            start=start,
            resolution=timedelta(minutes=15),
            values=[20.0 for _ in range(n)],
        ),
    )
    price_scenario_3 = Scenario(
        probability=0.3,
        value=TimeSeries(
            start=start,
            resolution=timedelta(minutes=15),
            values=[float(i) for i in range(n)],
        ),
    )
    return PriceScenarios(
        {1: price_scenario_1, 2: price_scenario_2, 3: price_scenario_3}
    )


def test_price_scenarios():
    price_scenarios = create_price_scenarios_with_probaility(start=None)

    _, values_dim2 = price_scenarios.values.shape
    (probabilities_dim1,) = price_scenarios.probabilities.shape

    assert values_dim2 == probabilities_dim1


def crate_model_builder() -> ModelBuilder:
    return ModelBuilder(
        battery=create_battery(),
        price_scenarios=create_price_scenarios_with_probaility(
            start=datetime(2020, 1, 1)
        ),
    )


def test_model_builder():
    model_builder = crate_model_builder()
    assert isinstance(model_builder, ModelBuilder)

    # Inner model builder parts
    assert isinstance(model_builder._model, linopy.Model)
    for var in [
        model_builder._var_level,
        model_builder._var_bid_quantity,
        model_builder._var_bid_accepted,
        model_builder._var_dispatch,
    ]:
        assert isinstance(var, linopy.Variable)
        assert var is model_builder._model.variables[var.name]


@pytest.fixture(scope="session")
def solution() -> Solution:
    return (
        crate_model_builder()
        .constrain_storage_level(soc_start=("==", 0.5), soc_end=("==", 0.5))
        .constrain_bidding_strategy()
        .constrain_imbalance()
        .add_objective(penalize_imbalance=1000.0)
        .solve()
    )


def test_model_solution(solution: Solution):
    assert isinstance(solution, Solution)


def test_model_quantity_balance(solution: Solution):
    model_builder = solution._model_builder

    imbalance = model_builder._var_imbalance.solution.sel(
        {model_builder._idx_position.name: Position.long.value}
    ) - model_builder._var_imbalance.solution.sel(
        {model_builder._idx_position.name: Position.short.value}
    )

    quantity_accepted = model_builder._var_bid_quantity_accepted.solution

    dispatch = model_builder._var_dispatch.solution

    print(model_builder._model.status)
    assert ((dispatch - quantity_accepted - imbalance) == 0.0).all().item()


def test_simultenous_charge_and_discharge():
    start = None
    n = 24

    battery = Battery(
        duration=timedelta(hours=2),
        power=2.0,
        efficiency=0.9,
        variable_cost=0,
    )

    price_scenario_1 = Scenario(
        probability=0.4,
        value=TimeSeries(
            start=start,
            resolution=timedelta(minutes=15),
            values=[sin(pi * 2 * i / 24) - 100 for i in range(n)],
        ),
    )
    price_scenario_2 = Scenario(
        probability=0.6,
        value=TimeSeries(
            start=start,
            resolution=timedelta(minutes=15),
            values=[20.0 for _ in range(n)],
        ),
    )

    price_scenarios = PriceScenarios({1: price_scenario_1, 2: price_scenario_2})

    solution = (
        ModelBuilder(battery=battery, price_scenarios=price_scenarios)
        .constrain_storage_level(soc_start=("==", 0.5), soc_end=("==", 0.5))
        .constrain_bidding_strategy()
        .constrain_imbalance()
        .add_objective(penalize_imbalance=1000.0)
        .solve()
    )

    assert solution._simultanous_actvity()

    solution = (
        ModelBuilder(battery=battery, price_scenarios=price_scenarios)
        .constrain_storage_level(soc_start=("==", 0.5), soc_end=("==", 0.5))
        .constrain_bidding_strategy()
        .constrain_simultanous_dispatch()
        .constrain_imbalance()
        .add_objective(penalize_imbalance=1000.0)
        .solve()
    )

    solution._model_builder._var_dispatch.solution.min("direction").max()

    assert not solution._simultanous_actvity()


if __name__ == "__main__":
    test_model_builder()
