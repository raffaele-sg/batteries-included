from datetime import datetime, timedelta
from math import pi, sin

import linopy

from batteries_included.model.common import (
    Battery,
    Parameters,
    State,
    TimeSeries,
)
from batteries_included.model.price_uncertainty import (
    ModelBuilder,
    PriceScenarios,
    Scenario,
)


def create_battery():
    return Battery(
        state=State(soc=0.5),
        parameters=Parameters(
            duration=timedelta(hours=2),
            power=2.0,
            efficiency=0.9,
        ),
    )


def create_price_scenarios():
    return TimeSeries(
        start=None,
        resolution=timedelta(minutes=15),
        values=[(sin(pi * 2 * i / 96), 20.0) for i in range(96)],
    )


def create_price_scenarios_with_probaility(start: None | datetime):
    n = 96
    price_scenario_1 = Scenario(
        probability=0.1,
        value=TimeSeries(
            start=start,
            resolution=timedelta(minutes=15),
            values=[sin(pi * 2 * i / 96) for i in range(n)],
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


def test_model_builder():
    mb = ModelBuilder(
        battery=create_battery(),
        price_scenarios=create_price_scenarios_with_probaility(
            start=datetime(2020, 1, 1)
        ),
    )
    assert isinstance(mb._model, linopy.Model)

    for var in [
        mb._var_level,
        mb._var_bid_quantity,
        mb._var_bid_price,
        mb._var_bid_accepted,
        mb._var_dispatch,
    ]:
        assert isinstance(var, linopy.Variable)
        assert var is mb._model.variables[var.name]

    mb.constrain_storage_level()
    mb.constrain_storage_level_end(0.5)
    mb.add_objective()
    mb.solve()


if __name__ == "__main__":
    test_model_builder()
