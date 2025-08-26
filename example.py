from datetime import datetime, timedelta

from batteries_included.model.common import (
    Battery,
    PriceScenarios,
    Scenario,
    TimeSeries,
)
from batteries_included.model.optimization import ModelBuilder

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
        # variation (4)
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

battery = Battery(
    duration=timedelta(hours=2.0),
    power=2.0,  # MW
    efficiency=0.9,
)

# Model one dispatch strategy for all scenarios
solution_dispatch_unique = (
    ModelBuilder(
        battery=battery,
        price_scenarios=price_scenarios,
    )
    .constrain_storage_level(soc_start=("==", 0.5), soc_end=(">=", 0.5))
    .contain_dispatch_across_scenarios()
    .constrain_imbalance()
    .add_objective(penalize_imbalance=1000.0)
    .solve()
)
print("- Objective (dispatch_unique)", solution_dispatch_unique.objective())

# Model multiple independent dispatch strategies (unconstrained)
solution_dispatch_unconstrained = (
    ModelBuilder(
        battery=battery,
        price_scenarios=price_scenarios,
    )
    .constrain_storage_level(soc_start=("==", 0.5), soc_end=(">=", 0.5))
    .constrain_imbalance()
    .add_objective(penalize_imbalance=1000.0)
    .solve()
)
print(
    "- Objective (dispatch_unconstrained)", solution_dispatch_unconstrained.objective()
)


# Model multiple dependent dispatch strategies (backed by a price bidding strategy)
solution_optimal_bidding = (
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
print("- Objective (optimal_bidding)", solution_optimal_bidding.objective())
