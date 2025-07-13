from src.model import Battery, BatteryParameters, BatteryState


def test_example():
    for i in Battery, BatteryParameters, BatteryState:
        example = i.example()
        assert isinstance(example, i)
