from batteries_included.model.common import Battery


def test_example():
    example = Battery.example()
    assert isinstance(example, Battery)
