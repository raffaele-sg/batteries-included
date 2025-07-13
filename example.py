from batteries_included.model import Battery, TimeSeries
from batteries_included.visualization import show_in_terminal

price = TimeSeries.example()
battery = Battery.example()

show_in_terminal(battery=battery, price=price)
