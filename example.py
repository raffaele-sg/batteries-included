from datetime import timedelta

from batteries_included.model import (
    Battery,
    Parameters,
    State,
    TimeSeries,
)
from batteries_included.optimization import SimulationManager, Variables

# from batteries_included.visualization import show_in_terminal
# price = TimeSeries.example()
# battery = Battery.example()
# show_in_terminal(battery=battery, price=price)

# Define a price series
price = TimeSeries(
    start=None,
    resolution=timedelta(minutes=15),
    values=[0.12, 0.11, 0.10, 0.10, 0.10, 0.11, 0.13, 0.16],  # EUR per kWh
)


# Define a battery consisting of a state and a set of technical paramenters
battery = Battery(
    state=State(soc=0.5),
    parameters=Parameters(
        duration=timedelta(hours=2.0),
        power=2.0,  # kW
        efficiency=0.9,
    ),
)


# Access the result of a model that is built solved under the hood
extractor = SimulationManager.from_inputs(battery=battery, price=price)

level = extractor.to_timeseries(variable=Variables.level)
buy = extractor.to_timeseries(variable=Variables.buy)
sell = extractor.to_timeseries(variable=Variables.sell)

print("\n Storage level [kWh]: \n ->", level)
print("\n Power used for charging [kW]: \n ->", buy)
print("\n Power delivered by discharing [kW]: \n ->", sell)
