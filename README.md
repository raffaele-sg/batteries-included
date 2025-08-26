# Scope
This repository provides a Python package that finds the optimial (feasible) bidding strategy for a battery facing price uncertainty. 

Although the application goes beyond a "simple" LP optimization to determine the dispach with perfect price foresight, the main goal was to experiment with new tools and design patterns.

That said, if you're looking for a minimal package that solves this exact problem, feel free to use it.

## Why is this interesting?

The Day-Ahead electricity market in Europe is organized as a pay-as-clear auction. This means that every accepted bid is settled at the market’s clearing price. For a battery bidding is not trivial because accepted buy bids translate into charging, accepted sell bids translate into discharging, and the battery must respect its physical constraints.

Whether a bid is accepted depends on its bid price compared to the auction clearing price, which is uncertain in advance. How do we find a suitable bidding strategy when the clearing price is uncertain?  

If we represent the uncertainty of market prices with a set of representative scenarios (each with an associated probability), then the optimization problem becomes: _**Choose one bidding strategy** (a set of price-quantity offers), such that it **maximizes expected profit** across all scenarios, while ensuring that the implied dispatch is **physically feasible** in each individual scenario._

## Example

Take the exmple below showing three price scenarios (and the optimized bidding strategy). The proposed approach differs from two common benchmarks. On the one hand, **optimizing dispatch individually per scenario** (perfect foresight) gives the highest possible profit (472 €), but it is infeasible since only one bidding strategy can be submitted. On the other hand, **optimizing a single deterministic dispatch** for underestimates the potential profits (358 €). By contrast, **optimizing the bidding strategy across scenarios** yields the best feasible outcome under market rules (406 €).


# Quickstart
## Installation
```bash
pip install git+https://github.com/raffaele-sg/batteries-included.git@v0.1.0-alpha
```

## Use

```python
from datetime import timedelta

from batteries_included.model import (
    Battery,
    Parameters,
    State,
    TimeSeries,
)
from batteries_included.optimization import Extractor, Variables


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


# Access the result of a model that is built and solved under the hood
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

# Formulation

This repository implements an optimization model for **battery bidding in electricity markets under price uncertainty**.  
The model is a **stochastic mixed-integer linear program (MILP)** using [linopy](https://github.com/PyPSA/linopy).

## Nomenclature
| <div style="width:290px">Sets and Indices</div>   |<div style="width:290px">Sets and Indices</div>|
| -------------------------------------             | -|
| $t \in T$                                         | time steps
| $s \in S$                                         | price scenarios
| $d \in \{\text{buy}, \text{sell}\}$               | bid direction  
| $p \in \{\text{long}, \text{short}\}$             | imbalance position

| <div style="width:290px">Parameters</div>         | <div style="width:290px">Sets and Indices</div>|
| -                                                 | -|
| $P^{\max}$                                        | charging/discharging power limit  
| $E^{\max}$                                        | energy capacity  
| $\eta^{\text{ch}}, \eta^{\text{dis}}$             | charging/discharging efficiencies  
| $\pi_{t,s}$                                       | market price at time $t$ in scenario $s$  
| $\rho_s$                                          | probability of scenario $s$  
| $\Delta t$                                        | time-step duration (hours)  
| $\lambda$                                         | imbalance penalty  


| <div style="width:290px">Decision Variables</div> | <div style="width:290px">Sets and Indices</div>|
| -                                                 | -|
|$\text{dispatch}_{t,d,s} \in [0, P^{\max}]$        | actual power dispatched (buy/sell)  
|$\text{level}_{t,s} \in [0, E^{\max}]$             | state of charge  
|$q_{t,d} \in [0, P^{\max}]$                        | bid quantity  
|$q^{\text{acc}}_{t,d,s}$                           | accepted bid quantity  
|$p_{t,d}$                                          | bid price  
|$a_{t,d,s} \in \{0,1\}$                            | bid acceptance indicator  
|$\text{imbalance}_{t,d,s,p} \ge 0$                 | imbalance variables  

## Storage Dynamics
For all $t \in T, s \in S$:
$$
\text{level}_{t,s} - \text{level}_{t-1,s}
= \Delta t \left( \eta^{\text{ch}} \cdot \text{dispatch}_{t,\text{buy},s}
- \frac{1}{\eta^{\text{dis}}} \cdot \text{dispatch}_{t,\text{sell},s} \right),
$$
with initialization via $\text{level}^{\text{init}}$ and a disired final state.


## Bid Acceptance Logic

The accepted bid quantity should equal the offered bid quantity if the bid is accepted, and zero otherwise:

$$
q^{\text{acc}}_{t,d,s} = q_{t,d} \cdot a_{t,d,s}.
$$

This is a **bilinear relation**, which is not directly supported in a linear MILP solver. To preserve linearity, we reformulate it using the **Big-M method**. Let $M = P^{\max}$ (the maximum possible bid quantity). Then the following set of linear inequalities ensures equivalence:
$$
\begin{aligned}
& q^{\text{acc}}_{t,d,s} \ge -M \cdot a_{t,d,s} \\
& q^{\text{acc}}_{t,d,s} \le M \cdot a_{t,d,s} \\
& q^{\text{acc}}_{t,d,s} \ge q_{t,d} - M \cdot (1 - a_{t,d,s}) \\
& q^{\text{acc}}_{t,d,s} \le q_{t,d} + M \cdot (1 - a_{t,d,s})
\end{aligned}
$$

Interpretation:
- If $a_{t,d,s} = 0$ (bid not accepted), then the constraints enforce $q^{\text{acc}}_{t,d,s} = 0$.  
- If $a_{t,d,s} = 1$ (bid accepted), then the constraints enforce $q^{\text{acc}}_{t,d,s} = q_{t,d}$.  


## Bid Price Consistency (Monotonicity)

Bid acceptance must follow a **monotonicity rule**. Scenarios are sorted by increasing price $\pi_{t,s}$ at time $t$ and direction $d$. For consecutive scenarios $s_\text{low}$ (lower price) and $s_\text{high}$ (higher price), the following constraints are imposed:

- **Buy bids:** if the bid is accepted at a *higher* price, then it must also be accepted at any *lower* price ($ a_{t,\text{buy},s_\text{low}} \ge a_{t,\text{buy},s_\text{high}} $).
- **Sell bids:** if the bid is accepted at a *lower* price, then it must also be accepted at any *higher* price ($ a_{t,\text{sell},s_\text{low}} \le a_{t,\text{sell},s_\text{high}}$).


## Imbalance Constraints
Deviations between dispatch and accepted bids:
$$
\text{imbalance}_{t,d,s,\text{long}} - \text{imbalance}_{t,d,s,\text{short}}
= \text{dispatch}_{t,d,s} - q^{\text{acc}}_{t,d,s}.
$$

## Objective Function
Maximize expected market profit minus expected imbalance penalties:
$$
\max \;
\Delta t \sum_{t,s} \rho_s \left(
q^{\text{acc}}_{t,\text{sell},s}\,\pi_{t,s}
- q^{\text{acc}}_{t,\text{buy},s}\,\pi_{t,s}
\right)
- \lambda \sum_{t,d,s,p} \Delta t \,\rho_s \,\text{imbalance}_{t,d,s,p}.
$$
