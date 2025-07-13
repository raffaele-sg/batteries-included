from __future__ import annotations

import xarray

from src.model import Battery, TimeSeries
from src.optimization import build_dispatch


def main():
    model = build_dispatch(battery=Battery.example(), price=TimeSeries.example())
    print(model.status)
    model.solve(output_flag=False)
    print(model.status)

    x: xarray.Dataset = model.solution
    print(type(x))
    print(x.get("level"))


if __name__ == "__main__":
    main()
