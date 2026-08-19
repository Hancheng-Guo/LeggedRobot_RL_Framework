from envs.tasks.managers.observation.terms.base import BaseObservationTerm
from utils.string import camel_to_snake


OBSERVATION_CLASS_MAP: dict[str, type[BaseObservationTerm]] = {}


def register_observation(
    cls: type[BaseObservationTerm],
) -> type[BaseObservationTerm]:

    name = camel_to_snake(cls.__name__)

    if name in OBSERVATION_CLASS_MAP:
        raise ValueError(
            f"Observation term '{name}' is already registered."
        )

    OBSERVATION_CLASS_MAP[name] = cls
    
    return cls


def get_observation_class(
    name: str,
) -> type[BaseObservationTerm]:

    if name not in OBSERVATION_CLASS_MAP:
        raise ValueError(
            f"Unknown observation term '{name}'."
        )

    return OBSERVATION_CLASS_MAP[name]
