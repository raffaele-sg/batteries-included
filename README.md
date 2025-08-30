# Scope
This repository provides a Python (3.13) package for finding an optimal and feasible bidding strategy for a battery under price uncertainty.

This project started as an experiment with new tools and design patterns. It began with a minimal LP formulation for battery dispatch, and gradually evolved to address price uncertainty, which proved to be the more interesting methodological challenge. The package remains an experimental tool developed for my own research. That said, if it happens to solve exactly the problem you are interested in, feel free to try it out and experiment with it.  


## Why is this interesting?

The Day-Ahead electricity market in Europe is organized as a pay-as-clear auction. This means that every accepted bid is settled at the market’s clearing price. For a battery bidding is not trivial because accepted buy bids translate into charging, accepted sell bids translate into discharging, and the battery must respect its physical constraints.

Whether a bid is accepted depends on its bid price compared to the auction clearing price, which is uncertain in advance. How do we find a suitable bidding strategy when the clearing price is uncertain?  

If we represent the uncertainty of market prices with a set of representative scenarios (each with an associated probability), then the optimization problem becomes:

_**Choose one bidding strategy** (a set of price-quantity offers), such that it **maximizes expected profit** across all scenarios, while ensuring that the implied dispatch is physically feasible in **each individual scenario**._

## Example

Take the exmple below showing three price scenarios (and the optimized bidding strategy). The proposed approach differs from two common benchmarks. On the one hand, **optimizing dispatch individually per scenario** (perfect foresight) gives the highest possible profit (472 €), but it is infeasible since only one bidding strategy can be submitted. On the other hand, **optimizing a single deterministic dispatch** for underestimates the potential profits (358 €). By contrast, **optimizing the bidding strategy across scenarios** yields the best feasible outcome under market rules (406 €).

![price](https://github.com/user-attachments/assets/ab91b084-ba13-4594-918b-217e9992e1b7)

# Quickstart

### Installation
```bash
pip install git+https://github.com/raffaele-sg/batteries-included.git@v1.0.0-alpha
```

The package can be installed directly from GitHub using the command above. This installation method is intended for experimentation and research purposes, and is **not suited for production systems**. If you are interested in using this code in a production environment, please reach out to discuss a suitable distribution and support.  

### Price scenario definition
```python
from datetime import datetime, timedelta
from batteries_included.model.common import PriceScenarios, Scenario, TimeSeries


start = datetime(1981, 9, 21)
resolution = timedelta(hours=1)

# fmt: off
price_scenarios = PriceScenarios(
    scenarios={
        "Scenario 1": Scenario(
            probability=1/3,
            value=TimeSeries(
                start=start,
                resolution=resolution,
                values=[
                     97.70,  93.18,  87.64,  85.78,  89.66,  98.80, 108.84, 110.12, 
                    103.91,  90.05,  80.98,  69.06,  49.07,  35.00,  42.03,  73.02,
                     81.45, 101.69, 109.16, 135.15, 126.11, 114.95, 103.49,  97.34,
                ],
            ),
        ),
        "Scenario 2": Scenario(
            probability=1/3,
            value=TimeSeries(
                start=start,
                resolution=resolution,
                values=[
                     83.97,  72.65,  64.62,  68.07,  62.18,  61.60,  55.79,  46.43, 
                     37.61,  47.06,  34.63,  37.15,  58.63,  51.89,  89.00, 112.05,
                     79.16, 100.55, 120.27, 165.49, 171.94, 144.63, 108.39, 108.09,
                ],
            ),
        ),
        "Scenario 3": Scenario(
            probability=1/3,
            value=TimeSeries(
                start=start,
                resolution=resolution,
                values=[
                    112.53,  95.98,  82.20,  73.79,  74.10,  83.36, 108.46, 114.15,
                    106.91,  77.74,  80.62,  45.24,  41.33,  24.16,   6.48,  44.97,
                     60.05,  67.50,  87.97, 146.32, 163.20, 166.38, 172.63, 151.31,
                ],
            ),
        ),
    }
)
# fmt: on
```
Scenarios can be defined by assigning each a probability (with all probabilities summing to 1), a start time, a time resolution, and a list of price values (in €/MWh). In the example above, three scenarios are provided with equal probability and hourly resolution. The resolution is flexible — for instance, it could be 15 minutes — and the model automatically accounts for this under the hood when calculating the dispatch.


### Battery definition
```python
from batteries_included.model.common import Battery


battery = Battery(
    duration=timedelta(hours=2.0),
    power=2.0,      # MW
    efficiency=0.9, # Round-trip
)
```

The battery is characterized by its duration (i.e. the time it can discharge at full power), its power capacity in MW, and its round-trip efficiency (the fraction of energy lost after a full charge–discharge cycle).


### Model definition and solution
```python
from batteries_included.model.optimization import ModelBuilder


solution = (
    ModelBuilder(
        battery=battery,
        price_scenarios=price_scenarios,
    )
    .constrain_storage_level(soc_start=("==", 0.5), soc_end=(">=", 0.5))
    .constrain_bidding_strategy()
    .constrain_imbalance()
    .add_objective(penalize_imbalance=1000.0)
    .solve()
)
```

The `ModelBuilder` brings together the battery and the price scenarios to formulate and solve the optimization problem. Constraints can be added step by step: in the example above, the storage level is fixed to start at 50% and end at least at 50%, a bidding strategy is enforced, and imbalances are allowed with a penalty cost. The method `.solve()` then computes the optimal bidding strategy and resulting dispatch.  

Importantly, the choice of constraints determines the type of benchmark being simulated:  
- Removing `.constrain_bidding_strategy()` simulates profits when dispatch is free to vary by scenario (perfect foresight).  
- Replacing `.constrain_bidding_strategy()` with `.constrain_dispatch_across_scenarios()` simulates one single deterministic dispatch across all scenarios (no bidding strategy).  

These variations correspond to the three cases illustrated in the **Example** section of the README.


### Extract the bidding strategy
```python
import pandas as pd


df = pd.DataFrame(
    solution.collect_bids(
        margin=10.0,
        remove_quantity_bids_below=0.0001,
    )
)
```

The resulting bids can be collected into a DataFrame using the `.collect_bids()` method. The bid price is determined by analyzing the scenarios in which the bid is accepted versus rejected: for example, if a sell bid is accepted whenever the price is above 10 and rejected when it is below 20, the bid price will be set as the average of those thresholds. If a bid is accepted in all scenarios, the price is set to the highest scenario price plus the specified margin. Conversely, if the bid quantity is zero, the price is undefined. To handle this, bids with very small quantities (below a user-defined threshold) are filtered out.  

The resulting dataframe should look something like this: 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>direction</th>
      <th>quantity</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1981-09-21 00:00:00</td>
      <td>BidDirection.sell</td>
      <td>1.897367</td>
      <td>73.970</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981-09-21 02:00:00</td>
      <td>BidDirection.buy</td>
      <td>0.216370</td>
      <td>97.640</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1981-09-21 03:00:00</td>
      <td>BidDirection.buy</td>
      <td>2.000000</td>
      <td>95.780</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981-09-21 04:00:00</td>
      <td>BidDirection.buy</td>
      <td>2.000000</td>
      <td>99.660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981-09-21 06:00:00</td>
      <td>BidDirection.sell</td>
      <td>2.000000</td>
      <td>45.790</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981-09-21 07:00:00</td>
      <td>BidDirection.sell</td>
      <td>1.794733</td>
      <td>36.430</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1981-09-21 08:00:00</td>
      <td>BidDirection.buy</td>
      <td>0.216370</td>
      <td>70.760</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1981-09-21 10:00:00</td>
      <td>BidDirection.buy</td>
      <td>2.000000</td>
      <td>57.625</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1981-09-21 11:00:00</td>
      <td>BidDirection.buy</td>
      <td>2.000000</td>
      <td>41.195</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1981-09-21 12:00:00</td>
      <td>BidDirection.buy</td>
      <td>0.216370</td>
      <td>53.850</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1981-09-21 12:00:00</td>
      <td>BidDirection.sell</td>
      <td>1.800000</td>
      <td>53.850</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1981-09-21 13:00:00</td>
      <td>BidDirection.buy</td>
      <td>2.000000</td>
      <td>61.890</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1981-09-21 14:00:00</td>
      <td>BidDirection.buy</td>
      <td>2.000000</td>
      <td>65.515</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1981-09-21 19:00:00</td>
      <td>BidDirection.sell</td>
      <td>1.794733</td>
      <td>125.150</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1981-09-21 20:00:00</td>
      <td>BidDirection.sell</td>
      <td>2.000000</td>
      <td>116.110</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1981-09-21 22:00:00</td>
      <td>BidDirection.buy</td>
      <td>0.108185</td>
      <td>182.630</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1981-09-21 23:00:00</td>
      <td>BidDirection.buy</td>
      <td>2.000000</td>
      <td>161.310</td>
    </tr>
  </tbody>
</table>


# Formulation Overview

The model is formulated as a a **stochastic mixed-integer linear program (MILP)** using [linopy](https://github.com/PyPSA/linopy).

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
For all $`t \in T, s \in S`$:

```math
\text{level}_{t,s} - \text{level}_{t-1,s} = \Delta t \left( \eta^{\text{ch}} \cdot \text{dispatch}_{t,\text{buy},s} - \frac{1}{\eta^{\text{dis}}} \cdot \text{dispatch}_{t,\text{sell},s} \right),
```
with initialization via $\text{level}^{\text{init}}$ and a disired final state.


## Bid Acceptance Logic

The accepted bid quantity should equal the offered bid quantity if the bid is accepted, and zero otherwise:

```math
q^{\text{acc}}_{t,d,s} = q_{t,d} \cdot a_{t,d,s}.
```

This is a **bilinear relation**, which is not directly supported in a linear MILP solver. To preserve linearity, we reformulate it using the **Big-M method**. Let $M = P^{\max}$ (the maximum possible bid quantity). Then the following set of linear inequalities ensures equivalence:
```math
\begin{aligned}
& q^{\text{acc}}_{t,d,s} \ge -M \cdot a_{t,d,s} \\
& q^{\text{acc}}_{t,d,s} \le M \cdot a_{t,d,s} \\
& q^{\text{acc}}_{t,d,s} \ge q_{t,d} - M \cdot (1 - a_{t,d,s}) \\
& q^{\text{acc}}_{t,d,s} \le q_{t,d} + M \cdot (1 - a_{t,d,s})
\end{aligned}
```

Interpretation:
- If $`a_{t,d,s} = 0`$ (bid not accepted), then the constraints enforce $`q^{\text{acc}}_{t,d,s} = 0`$.  
- If $`a_{t,d,s} = 1`$ (bid accepted), then the constraints enforce $`q^{\text{acc}}_{t,d,s} = q_{t,d}`$.  


## Bid Price Consistency (Monotonicity)

Bid acceptance must follow a **monotonicity rule**. Scenarios are sorted by increasing price $\pi_{t,s}$ at time $t$ and direction $d$. For consecutive scenarios $`s_\text{low}`$ (lower price) and $`s_\text{high}`$ (higher price), the following constraints are imposed:

- **Buy bids:** if the bid is accepted at a *higher* price, then it must also be accepted at any *lower* price ($`a_{t,\text{buy},s_\text{low}} \ge a_{t,\text{buy},s_\text{high}}`$).
- **Sell bids:** if the bid is accepted at a *lower* price, then it must also be accepted at any *higher* price ($`a_{t,\text{sell},s_\text{low}} \le a_{t,\text{sell},s_\text{high}}`$).


## Imbalance Constraints
Deviations between dispatch and accepted bids:

```math
\text{imbalance}_{t,d,s,\text{long}} - \text{imbalance}_{t,d,s,\text{short}}
= \text{dispatch}_{t,d,s} - q^{\text{acc}}_{t,d,s}.
```

## Objective Function
Maximize expected market profit minus expected imbalance penalties:

```math
\max \;
\Delta t \sum_{t,s} \rho_s \left(
q^{\text{acc}}_{t,\text{sell},s} \pi_{t,s}
- q^{\text{acc}}_{t,\text{buy},s} \pi_{t,s}
\right)
- \lambda \sum_{t,d,s,p} \Delta t \,\rho_s \,\text{imbalance}_{t,d,s,p}.
```
