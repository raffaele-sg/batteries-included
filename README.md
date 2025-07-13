# Scope
This repository provides a lightweight Python package that optimizes battery dispatch based on an exogenous price profile. The problem formulation is deliberately simple and standard, as the main goal was to experiment with new tools and design patterns. That said, if you're looking for a minimal package that solves this exact problem, feel free to use it.

# Quickstart
## Usage

```python
from datetime import timedelta

from batteries_included.model import (
    Battery,
    Parameters,
    State,
    TimeSeries,
)
from batteries_included.optimization import Extractor, Variables

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
extractor = Extractor.from_inputs(battery=battery, price=price)

level = extractor.to_timeseries(variable=Variables.level)
buy = extractor.to_timeseries(variable=Variables.buy)
sell = extractor.to_timeseries(variable=Variables.sell)

print("\n Storage level [kWh]: \n ->", level)
print("\n Power used for charging [kW]: \n ->", buy)
print("\n Power delivered by discharing [kW]: \n ->", sell)
```

This should print:
```console
Storage level [kWh]: 
-> TimeSeries(start=None, resolution=datetime.timedelta(seconds=900), values=[1.6310676063100003, 1.6310676063100003, 2.105409255336, 2.579750904362, 3.054092553388, 3.054092553388, 2.527046276694, 2.0])

Power used for charging [kW]: 
-> TimeSeries(start=None, resolution=datetime.timedelta(seconds=900), values=[0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0])

Power delivered by discharing [kW]: 
-> TimeSeries(start=None, resolution=datetime.timedelta(seconds=900), values=[1.4000000000159372, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0])
```

Note that: 
- The initial level is 2.00 kWh because of the following inputs:
    - Initial 0.5 SoC
    - 2 h duration
    - 2.0 kW power
- The power that is delivered in the first interval is 1.40 kW:
    - This corresponts to 0.35 kWh because the time resolution is 15 minutes
    - The storage level drops by 0.37 kWh becuase of the input round-trip efficiency (0.35 kWh / 0.9^0.5) 
- The storage level after the first interval is in fact 1.63 kWh, i.e. 2.00 - 0.37 kWh


# Development
## Try as module user
Install as editable:
```bash
uv pip install -e . ".[dev]"
```

Run the example.py file:
```bash
uv run example.py
```

Runs test:
```bash
uv run pytest
```
