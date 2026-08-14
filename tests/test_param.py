import pytest

from utils.param import update_attributes


class InstanceForTest:

    def __init__(self):
        self.name = "old"
        self.value = 10
        self.empty = None


def test_update_attributes():

    instance = InstanceForTest()

    update_attributes(
        instance,
        name="new",
        value=20,
    )

    assert instance.name == "new"
    assert instance.value == 20


def test_update_attributes_keep_existing_value():

    instance = InstanceForTest()

    update_attributes(
        instance,
        name=None,
        value=None,
    )

    assert instance.name == "old"
    assert instance.value == 10


def test_update_attributes_missing_value():

    instance = InstanceForTest()

    with pytest.raises(ValueError):
        update_attributes(
            instance,
            empty=None,
        )


def test_update_attributes_missing_attribute():

    instance = InstanceForTest()

    with pytest.raises(ValueError):
        update_attributes(
            instance,
            unknown=10,
        )