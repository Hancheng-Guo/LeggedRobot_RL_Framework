import pytest

from utils.param import update_attributes


class DummyInstance:

    def __init__(self) -> None:
        self.a = 1
        self.b = 2
        self.c = None


def test_update_attributes():

    instance = DummyInstance()

    update_attributes(
        instance,
        a=10,
        b=None,
    )

    assert instance.a == 10
    assert instance.b == 2


def test_update_attributes_missing_value():

    instance = DummyInstance()

    with pytest.raises(ValueError):
        update_attributes(
            instance,
            c=None,
        )


def test_update_attributes_unknown_attribute():

    instance = DummyInstance()

    with pytest.raises(ValueError):
        update_attributes(
            instance,
            unknown=None,
        )