from abc import ABC, abstractmethod
from typing import Any


class BaseSimulator(ABC):

    def __init__(self) -> None:
        pass


    @abstractmethod
    def reset(
        self,
        env_ids: Any = None,
    ) -> None:
        """
        Reset simulator state.
        """
        pass


    @abstractmethod
    def step(
        self,
        action,
    ) -> None:
        """
        Advance simulation by one environment step.
        """
        pass


    @abstractmethod
    def render(
        self,
    ) -> None:
        """
        Render simulation.
        """
        pass


    @abstractmethod
    def close(
        self,
    ) -> None:
        """
        Release simulator resources.
        """
        pass


# class BatchProxy:

#     def __init__(self, objects: Iterable[Any]) -> None:
#         object.__setattr__(
#             self,
#             "_objects",
#             list(objects),
#         )


#     def __getattr__(self, name: str):

#         values = [
#             getattr(obj, name)
#             for obj in self._objects
#         ]

#         if all(callable(value) for value in values):

#             def method(*args, **kwargs):
#                 return [
#                     value(*args, **kwargs)
#                     for value in values
#                 ]

#             return method

#         return BatchProxy(values)


#     def __setattr__(
#         self,
#         name: str,
#         value: Any,
#     ) -> None:

#         if name == "_objects":
#             object.__setattr__(
#                 self,
#                 name,
#                 value,
#             )
#             return

#         for obj in self._objects:
#             setattr(obj, name, value)


#     def __getitem__(
#         self,
#         index,
#     ):
#         return self._objects[index]


#     def __len__(self) -> int:
#         return len(self._objects)


#     def __iter__(self):
#         return iter(self._objects)