import plotext as plt

from batteries_included.model import Battery, TimeSeries
from batteries_included.optimization import Extractor, Variables


def show_in_terminal(battery: Battery, price: TimeSeries):
    extractor = Extractor.from_inputs(
        battery=battery,
        price=price,
    )

    plt.subplots(2, 1)

    plt.subplot(1, 1)
    plt.title("Battery dispatch")
    plt.bar(extractor.to_numpy(variable=Variables.buy).round(5), label="buy")
    plt.bar(-extractor.to_numpy(variable=Variables.sell).round(5), label="sell")
    plt.plot(
        extractor.to_numpy(variable=Variables.level).round(5),
        label="level",
        color="black",
    )

    plt.subplot(2, 1)
    plt.title("Electricity price")
    plt.plot(price.values, label="price")

    plt.show()
