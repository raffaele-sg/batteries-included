from datetime import timedelta
from math import pi, sin

import linopy
import numpy as np

from main import (
    Battery,
    BatteryParameters,
    BatteryState,
    Extractor,
    TimeSeries,
    Variables,
    build_dispatch,
)


def create_battery():
    return Battery(
        BatteryState(0.5),
        parameters=BatteryParameters(
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


def test_build_dispatch():
    model = build_dispatch(battery=create_battery(), price=create_prices())
    assert isinstance(model, linopy.Model)


def test_extractor():
    model = build_dispatch(battery=create_battery(), price=create_prices())
    model.solve(output_flag=False)
    extractor = Extractor(model)

    for variable in Variables:
        array = extractor.to_numpy(variable=variable)
        assert isinstance(array, np.ndarray)
