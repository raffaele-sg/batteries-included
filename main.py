from src.model import Battery, TimeSeries
from src.visualization import show_in_terminal


def main():
    price = TimeSeries.example()
    battery = Battery.example()

    show_in_terminal(battery=battery, price=price)


if __name__ == "__main__":
    main()
