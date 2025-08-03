from batteries_included.model.common import Battery, Parameters, State


def test_example():
    for i in Battery, Parameters, State:
        example = i.example()
        assert isinstance(example, i)
