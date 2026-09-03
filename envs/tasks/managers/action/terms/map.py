import torch

from envs.tasks.managers.action.terms.base import BaseActionTerm
from envs.tasks.managers.action.terms.registry import register_action
from envs.simulators.utils.context import ModelContext


@register_action
class LinearMap(BaseActionTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.input_dim = self.output_dim

        ctrl_range = model_context.actuator_ctrl_range
        self.center = 0.5 * (ctrl_range[:, 0] + ctrl_range[:, 1])
        self.scale = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])


    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:

        return self.center + value * self.scale


@register_action
class TanhMap(BaseActionTerm):

    def __init__(
        self,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.input_dim = self.output_dim


    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:

        return torch.tanh(value)


@register_action
class KeyframeCenteredLinearMap(BaseActionTerm):

    def __init__(
        self,
        model_context: ModelContext,
        *args, **kwargs,
    ) -> None:

        super().__init__(*args, **kwargs)

        self.input_dim = self.output_dim

        ctrl_range = model_context.actuator_ctrl_range
        center = model_context.actuator_default_ctrl

        if ctrl_range.shape != (self.output_dim, 2):
            raise ValueError(
                "'actuator_ctrl_range' must have shape "
                f"({self.output_dim}, 2), got {tuple(ctrl_range.shape)}."
            )
        if center.shape != (self.output_dim,):
            raise ValueError(
                "'actuator_default_ctrl' must have shape "
                f"({self.output_dim},), got {tuple(center.shape)}."
            )

        self.lower = ctrl_range[:, 0]
        self.upper = ctrl_range[:, 1]
        self.center = center

        if torch.any(self.lower > self.upper):
            raise ValueError("Actuator control lower limits must not exceed upper limits.")
        if torch.any((self.center < self.lower) | (self.center > self.upper)):
            raise ValueError(
                "Keyframe actuator controls must lie within actuator control limits."
            )

        self.negative_scale = self.center - self.lower
        self.positive_scale = self.upper - self.center


    def process(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:

        scale = torch.where(
            value < 0,
            self.negative_scale,
            self.positive_scale,
        )
        return self.center + value * scale
