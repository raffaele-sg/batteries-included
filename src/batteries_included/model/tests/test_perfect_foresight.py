from datetime import timedelta
from math import pi, sin

import linopy
import numpy as np

from batteries_included.model.common import (
    Battery,
    Parameters,
    State,
    TimeSeries,
)
from batteries_included.model.perfect_foresight import (
    Metadata,
    SimulationManager,
    Variables,
    build_dispatch,
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


def create_prices():
    return TimeSeries(
        start=None,
        resolution=timedelta(minutes=15),
        values=[sin(pi * 2 * i / 96) for i in range(96)],
    )


def create_price_scenarios():
    return TimeSeries(
        start=None,
        resolution=timedelta(minutes=15),
        values=[(sin(pi * 2 * i / 96), 20.0) for i in range(96)],
    )


def test_build_dispatch():
    metadata, model = build_dispatch(battery=create_battery(), price=create_prices())

    assert isinstance(metadata, Metadata)
    assert isinstance(model, linopy.Model)


def test_extractor():
    metadata, model = build_dispatch(battery=create_battery(), price=create_prices())
    extractor = SimulationManager(model=model, metadata=metadata)

    for variable in Variables:
        array = extractor.to_numpy(variable=variable)
        assert isinstance(array, np.ndarray)


def test_extractor_from_inputs():
    extractor = SimulationManager.from_inputs(
        battery=create_battery(),
        price=create_prices(),
    )

    for variable in Variables:
        array = extractor.to_numpy(variable=variable)
        assert isinstance(array, np.ndarray)

        timeseries = extractor.to_timeseries(variable=variable)
        assert isinstance(timeseries, TimeSeries)
