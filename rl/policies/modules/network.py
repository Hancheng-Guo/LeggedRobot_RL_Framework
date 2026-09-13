import torch
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from rl.policies.modules.recurrent import StatefulGRU
from rl.policies.modules.registry import build_module


@dataclass(frozen=True, slots=True)
class _InputReference:
    source: str
    feature_slice: slice | None = None


@dataclass(slots=True)
class _ModuleSpec:
    name: str
    config: dict[str, Any]
    inputs: tuple[_InputReference, ...]
    inherit: bool
    artifact: Mapping[str, Any] | None


class ConfigurableNetwork(torch.nn.Sequential):

    _MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
    _WINDOWS_RESERVED_NAMES = frozenset({
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    })
    _INPUT_REFERENCE_PATTERN = re.compile(
        r"^(?P<source>[^\[\]]+?)(?:\[(?P<start>\d+):(?P<stop>\d+)\])?$"
    )

    _SHAPE_PRESERVING_TYPES = (
        torch.nn.ReLU,
        torch.nn.Tanh,
        torch.nn.ELU,
        torch.nn.GELU,
        torch.nn.SiLU,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.Identity,
    )

    def __init__(
        self,
        modules: Sequence[Mapping[str, Any]],
        variables: Mapping[str, Any] | None = None,
        module_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
        input_slices: Mapping[str, slice] | None = None,
    ) -> None:
        
        self._validate_module_configs(modules)
        if variables is not None and not isinstance(variables, Mapping):
            raise TypeError("Network 'variables' must be a mapping.")
        if input_slices is not None and not isinstance(input_slices, Mapping):
            raise TypeError("Network 'input_slices' must be a mapping.")
        if module_artifacts is not None and not isinstance(
            module_artifacts, Mapping
        ):
            raise TypeError("Network 'module_artifacts' must be a mapping.")

        self._inherit_names: set[str] = set()
        self._module_configs: dict[str, dict[str, Any]] = {}
        self._explicit_names: set[str] = set()
        self._input_sources: dict[str, tuple[_InputReference, ...]] = {}
        self._input_slices = dict(input_slices or {})
        self._validate_input_slices()
        self._output_features: dict[str, int] = {}
        resolved_variables = self._resolve_variables(variables)
        root_input_features = resolved_variables.get("obs_dim")
        if (
            not isinstance(root_input_features, int)
            or isinstance(root_input_features, bool)
            or root_input_features <= 0
        ):
            raise ValueError("Network requires a positive integer 'obs_dim'.")
        for field, field_slice in self._input_slices.items():
            if field_slice.stop > root_input_features:
                raise ValueError(
                    f"Observation field {field!r} exceeds obs_dim "
                    f"{root_input_features}."
                )
        self._root_input_features = root_input_features
        named_modules = self._build_modules(
            modules=modules,
            variables=resolved_variables,
            artifacts=dict(module_artifacts or {}),
        )
        super().__init__(named_modules)


    @staticmethod
    def _validate_module_configs(
        modules: Sequence[Mapping[str, Any]],
    ) -> None:
        if isinstance(modules, (str, bytes)) or not isinstance(
            modules, Sequence
        ):
            raise TypeError("Network 'modules' must be a sequence.")
        if not modules:
            raise ValueError("Network 'modules' must be a non-empty sequence.")
        if any(not isinstance(config, Mapping) for config in modules):
            raise TypeError("Every network module config must be a mapping.")


    def _validate_input_slices(self) -> None:
        for field, field_slice in self._input_slices.items():
            if not isinstance(field, str) or not field:
                raise TypeError("Observation field names must be non-empty strings.")
            if not isinstance(field_slice, slice):
                raise TypeError(
                    f"Observation field {field!r} must be represented by a slice."
                )
            if (
                not isinstance(field_slice.start, int)
                or isinstance(field_slice.start, bool)
                or not isinstance(field_slice.stop, int)
                or isinstance(field_slice.stop, bool)
                or field_slice.start < 0
                or field_slice.start >= field_slice.stop
                or field_slice.step not in (None, 1)
            ):
                raise ValueError(
                    f"Observation field {field!r} must use a non-empty, "
                    "nonnegative unit-step slice."
                )


    def _resolve_variables(
        self,
        variables: Mapping[str, Any] | None,
    ) -> dict[str, Any]:

        resolved = dict(variables or {})
        for field, field_slice in self._input_slices.items():
            field_dim = field_slice.stop - field_slice.start
            resolved.setdefault(field, field_dim)
            resolved[f"obs.{field}.shape"] = field_dim
            resolved[f"obs.{field}.shape[-1]"] = field_dim
        if "obs_dim" in resolved:
            resolved["obs.shape"] = resolved["obs_dim"]
            resolved["obs.shape[-1]"] = resolved["obs_dim"]
        if "action_dim" in resolved:
            resolved["action.shape"] = resolved["action_dim"]
            resolved["action.shape[-1]"] = resolved["action_dim"]
        return resolved


    def _build_modules(
        self,
        modules: Sequence[Mapping[str, Any]],
        variables: dict[str, Any],
        artifacts: Mapping[str, Mapping[str, Any]],
    ) -> OrderedDict[str, torch.nn.Module]:

        named_modules: OrderedDict[str, torch.nn.Module] = OrderedDict()
        if any(
            not isinstance(name, str) or not isinstance(artifact, Mapping)
            for name, artifact in artifacts.items()
        ):
            raise TypeError(
                "Module artifact names must be strings and values must be "
                "mappings."
            )
        for index, raw_config in enumerate(modules):
            spec = self._parse_module_spec(
                raw_config=raw_config,
                index=index,
                available_modules=set(named_modules),
                artifacts=artifacts,
            )
            self._input_sources[spec.name] = spec.inputs
            input_features = self._resolve_input_features(spec.name)
            module, module_config = self._instantiate_module(
                spec=spec,
                input_features=input_features,
                variables=variables,
                artifacts=artifacts,
            )
            named_modules[spec.name] = module
            self._register_module(
                spec=spec,
                module=module,
                module_config=module_config,
                input_features=input_features,
                variables=variables,
            )
        return named_modules


    def _parse_module_spec(
        self,
        raw_config: Mapping[str, Any],
        index: int,
        available_modules: set[str],
        artifacts: Mapping[str, Mapping[str, Any]],
    ) -> _ModuleSpec:

        config = dict(raw_config)
        explicit_name = config.pop("name", None)
        raw_inputs = config.pop("inputs", None)
        module_type_name = config.get("type")
        artifact = artifacts.get(str(module_type_name))
        name = explicit_name or (
            str(module_type_name) if artifact is not None else str(index)
        )
        inherit = config.pop("inherit", False)

        if (
            not isinstance(name, str)
            or self._MODULE_NAME_PATTERN.fullmatch(name) is None
            or name.upper() in self._WINDOWS_RESERVED_NAMES
            or name == "obs"
        ):
            raise ValueError(
                "Module 'name' must be a non-empty, file-safe string."
            )
        if name in available_modules:
            raise ValueError(f"Duplicate module name: {name!r}.")
        if not isinstance(inherit, bool):
            raise TypeError("Module 'inherit' must be a boolean.")

        inputs = self._parse_inputs(
            raw_inputs=raw_inputs,
            module_name=name,
            available_modules=available_modules,
        )
        if explicit_name is not None or artifact is not None:
            self._explicit_names.add(name)

        return _ModuleSpec(name, config, inputs, inherit, artifact)


    def _instantiate_module(
        self,
        spec: _ModuleSpec,
        input_features: int,
        variables: Mapping[str, Any],
        artifacts: Mapping[str, Mapping[str, Any]],
    ) -> tuple[torch.nn.Module, dict[str, Any]]:

        if spec.artifact is None:
            module_config = spec.config
        else:
            module_type_name = spec.config.get("type")
            unsupported_parameters = set(spec.config) - {
                "type",
                "in_features",
                "out_features",
            }
            if unsupported_parameters:
                raise ValueError(
                    f"Artifact module {module_type_name!r} cannot override "
                    "its saved structure. Unsupported parameters: "
                    f"{sorted(unsupported_parameters)}."
                )
            artifact_config = spec.artifact.get("config")
            if not isinstance(artifact_config, Mapping):
                raise TypeError(
                    f"Artifact {module_type_name!r} has no valid config."
                )
            module_config = dict(artifact_config)

        module = self._build_module(
            module_config,
            input_features=input_features,
            variables=variables,
            module_artifacts=artifacts,
            input_slices=(
                self._input_slices
                if self._uses_root_input(spec)
                else {}
            ),
        )
        if spec.artifact is not None:
            artifact_state = spec.artifact.get("state_dict")
            if not isinstance(artifact_state, Mapping):
                raise TypeError(
                    f"Artifact {spec.config.get('type')!r} has no state_dict."
                )
            module.load_state_dict(artifact_state)
            self._validate_artifact_interface(
                spec=spec,
                module=module,
                input_features=input_features,
                variables=variables,
            )
        return module, module_config


    def _validate_artifact_interface(
        self,
        spec: _ModuleSpec,
        module: torch.nn.Module,
        input_features: int,
        variables: Mapping[str, Any],
    ) -> None:
        declared_input = (
            self._resolve_feature_assertion(
                value=spec.config["in_features"],
                variables=variables,
                field="in_features",
                module_name=spec.name,
            )
            if "in_features" in spec.config
            else None
        )
        if declared_input is not None and declared_input != input_features:
            raise ValueError(
                f"Artifact module {spec.name!r} declares {declared_input} "
                f"input features, but its configured inputs provide "
                f"{input_features}."
            )

        actual_output = self._infer_output_features(
            module=module,
            input_features=input_features,
        )
        declared_output = (
            self._resolve_feature_assertion(
                value=spec.config["out_features"],
                variables=variables,
                field="out_features",
                module_name=spec.name,
            )
            if "out_features" in spec.config
            else None
        )
        if declared_output is not None and declared_output != actual_output:
            raise ValueError(
                f"Artifact module {spec.name!r} declares {declared_output} "
                f"output features, but the saved module produces "
                f"{actual_output}."
            )


    @staticmethod
    def _resolve_feature_assertion(
        value: Any,
        variables: Mapping[str, Any],
        field: str,
        module_name: str,
    ) -> int:
        resolved = (
            variables.get(value, value)
            if isinstance(value, str)
            else value
        )
        if (
            not isinstance(resolved, int)
            or isinstance(resolved, bool)
            or resolved <= 0
        ):
            raise ValueError(
                f"Artifact module {module_name!r} {field!r} must resolve "
                "to a positive integer."
            )
        return resolved


    def _uses_root_input(self, spec: _ModuleSpec) -> bool:
        if len(spec.inputs) != 1 or spec.inputs[0].feature_slice is not None:
            return False
        source = spec.inputs[0].source
        return source == "obs" or (
            source == "$previous" and not self._output_features
        )


    def _register_module(
        self,
        spec: _ModuleSpec,
        module: torch.nn.Module,
        module_config: Mapping[str, Any],
        input_features: int,
        variables: dict[str, Any],
    ) -> None:

        name = spec.name
        self._module_configs[name] = deepcopy(dict(module_config))
        self._validate_module_input_features(
            name=name,
            module=module,
            input_features=input_features,
        )
        output_features = self._infer_output_features(
            module=module,
            input_features=input_features,
        )
        if output_features <= 0:
            raise ValueError(
                f"Module {name!r} must produce a positive number of "
                "output features."
            )
        self._output_features[name] = output_features
        variables[f"{name}.out_features"] = output_features
        variables[f"{name}.shape"] = output_features
        variables[f"{name}.shape[-1]"] = output_features
        if spec.inherit:
            self._inherit_names.add(name)


    @property
    def output_features(self) -> int:
        final_name = next(reversed(self._modules))
        return self._output_features[final_name]


    def _resolve_input_features(
        self,
        module_name: str
    ) -> int:

        dimensions: list[int] = []
        for reference in self._input_sources[module_name]:
            source = reference.source
            if source in {"obs", "$previous"}:
                if source == "$previous" and self._output_features:
                    previous_name = next(reversed(self._output_features))
                    dimensions.append(self._output_features[previous_name])
                else:
                    dimensions.append(self._root_input_features)
            elif source.startswith("obs."):
                term_slice = self._input_slices[source.removeprefix("obs.")]
                dimensions.append(term_slice.stop - term_slice.start)
            elif source in self._output_features:
                dimensions.append(self._output_features[source])
            else:
                raise ValueError(
                    f"Cannot resolve input features for module "
                    f"{module_name!r}: output features of {source!r} "
                    "are unknown."
                )
            if reference.feature_slice is not None:
                start = reference.feature_slice.start
                stop = reference.feature_slice.stop
                source_features = dimensions.pop()
                if stop > source_features:
                    raise ValueError(
                        f"Input slice {source}[{start}:{stop}] for module "
                        f"{module_name!r} exceeds its {source_features} "
                        "available features."
                    )
                dimensions.append(stop - start)
        return sum(dimensions)


    @staticmethod
    def _infer_output_features(
        module: torch.nn.Module,
        input_features: int,
    ) -> int:

        out_features = getattr(module, "out_features", None)
        if isinstance(out_features, int):
            return out_features
        if isinstance(module, StatefulGRU):
            return module.hidden_size
        if isinstance(module, ConfigurableNetwork):
            return module.output_features
        if isinstance(module, ConfigurableNetwork._SHAPE_PRESERVING_TYPES):
            return input_features
        raise ValueError(
            f"Cannot infer output features for module "
            f"{type(module).__name__!r}."
        )


    @staticmethod
    def _validate_module_input_features(
        name: str,
        module: torch.nn.Module,
        input_features: int,
    ) -> None:
        declared_input_features: int | None = None
        if isinstance(module, torch.nn.Linear):
            declared_input_features = module.in_features
        elif isinstance(module, StatefulGRU):
            declared_input_features = module.gru.input_size
        elif isinstance(module, ConfigurableNetwork):
            declared_input_features = module._root_input_features
        elif isinstance(module, torch.nn.LayerNorm):
            normalized_shape = module.normalized_shape
            if len(normalized_shape) == 1:
                declared_input_features = normalized_shape[0]

        if (
            declared_input_features is not None
            and declared_input_features != input_features
        ):
            raise ValueError(
                f"Module {name!r} expects {declared_input_features} input "
                f"features, but its configured inputs provide "
                f"{input_features}."
            )


    def _parse_inputs(
        self,
        raw_inputs: Any,
        module_name: str,
        available_modules: set[str],
    ) -> tuple[_InputReference, ...]:

        if raw_inputs is None:
            return (_InputReference("$previous"),)
        if isinstance(raw_inputs, str):
            sources = (raw_inputs,)
        elif isinstance(raw_inputs, Sequence):
            sources = tuple(raw_inputs)
        else:
            raise TypeError(f"Module {module_name!r} inputs must be strings.")
        if not sources or any(not isinstance(source, str) for source in sources):
            raise TypeError(f"Module {module_name!r} inputs must be strings.")
        references = tuple(
            self._parse_input_reference(source, module_name)
            for source in sources
        )
        for reference in references:
            source = reference.source
            if source in {"obs", "$previous"}:
                continue
            if source.startswith("obs."):
                field = source.removeprefix("obs.")
                if field not in self._input_slices:
                    raise KeyError(
                        f"Unknown observation field {field!r} for module "
                        f"{module_name!r}."
                    )
            elif source not in available_modules:
                raise KeyError(
                    f"Module {module_name!r} references unavailable input "
                    f"{source!r}. Inputs must refer to an earlier module."
                )
        return references


    @classmethod
    def _parse_input_reference(
        cls,
        source: str,
        module_name: str,
    ) -> _InputReference:
        match = cls._INPUT_REFERENCE_PATTERN.fullmatch(source)
        if match is None:
            raise ValueError(
                f"Invalid input reference {source!r} for module "
                f"{module_name!r}. Expected 'source' or 'source[start:stop]'."
            )
        start_text = match.group("start")
        stop_text = match.group("stop")
        if start_text is None or stop_text is None:
            return _InputReference(match.group("source"))
        start = int(start_text)
        stop = int(stop_text)
        if start >= stop:
            raise ValueError(
                f"Input slice for module {module_name!r} must have "
                "start less than stop."
            )
        return _InputReference(
            source=match.group("source"),
            feature_slice=slice(start, stop),
        )


    @staticmethod
    def _build_module(
        config: Mapping[str, Any],
        input_features: int,
        variables: Mapping[str, Any] | None,
        module_artifacts: Mapping[str, Mapping[str, Any]],
        input_slices: Mapping[str, slice],
    ) -> torch.nn.Module:

        module_config = dict(config)
        module_type_name = module_config.get("type")
        if module_type_name == "sequential":
            module_config.pop("type")
            nested_modules = module_config.pop("modules", None)
            if module_config:
                names = ", ".join(sorted(module_config))
                raise ValueError(
                    f"Unsupported sequential parameters: {names}."
                )
            if (
                isinstance(nested_modules, (str, bytes))
                or not isinstance(nested_modules, Sequence)
            ):
                raise TypeError("Sequential 'modules' must be a sequence.")
            nested_variables = dict(variables or {})
            nested_variables["obs_dim"] = input_features
            nested_variables["obs.shape"] = input_features
            nested_variables["obs.shape[-1]"] = input_features
            return ConfigurableNetwork(
                modules=nested_modules,
                variables=nested_variables,
                module_artifacts=module_artifacts,
                input_slices=input_slices,
            )
        return build_module(module_config, variables=variables)


    def export_module_artifacts(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "config": deepcopy(self._module_configs[name]),
                "state_dict": module.state_dict(),
            }
            for name, module in self.named_children()
            if name in self._explicit_names
        }


    def export_portable_module_artifacts(
        self,
    ) -> dict[str, dict[str, Any]]:
        artifacts = self.export_module_artifacts()
        for name, module in self.named_children():
            if not isinstance(module, ConfigurableNetwork):
                continue
            artifacts.update({
                f"{name}.{nested_name}": artifact
                for nested_name, artifact in (
                    module.export_portable_module_artifacts().items()
                )
            })
        return artifacts


    def inherit_modules_from(
        self,
        previous: "ConfigurableNetwork"
    ) -> None:

        previous_modules = dict(previous.named_children())
        for name in self._inherit_names:
            if name not in previous_modules:
                raise ValueError(
                    f"Cannot inherit module {name!r}: it did not exist in the "
                    "previous stage."
                )
            current_module = self.get_submodule(name)
            previous_module = previous_modules[name]
            if type(current_module) is not type(previous_module):
                raise TypeError(
                    f"Cannot inherit module {name!r}: module types differ."
                )
            current_state = current_module.state_dict()
            previous_state = previous_module.state_dict()
            if current_state.keys() != previous_state.keys() or any(
                current_state[key].shape != previous_state[key].shape
                for key in current_state
            ):
                raise ValueError(
                    f"Cannot inherit module {name!r}: parameter shapes differ."
                )
            current_module.load_state_dict(previous_state)


    @property
    def is_recurrent(self) -> bool:
        return any(
            isinstance(module, StatefulGRU)
            or isinstance(module, ConfigurableNetwork) and module.is_recurrent
            for module in self
        )


    def forward(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step(input)


    def forward_sequence(
        self,
        inputs: torch.Tensor,
        initial_state: dict[str, torch.Tensor],
        reset_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        if inputs.ndim < 3:
            raise ValueError(
                "Sequence input must have shape [time, batch, ...]."
            )
        if reset_mask.shape != inputs.shape[:2]:
            raise ValueError("reset_mask must have shape [time, batch].")
        if reset_mask.dtype != torch.bool:
            raise TypeError("reset_mask must have dtype torch.bool.")
        if inputs.shape[0] == 0:
            raise ValueError("Sequence input cannot be empty.")
        self._validate_root_input_features(inputs)

        self._validate_recurrent_state(initial_state)
        state = dict(initial_state)
        outputs: list[torch.Tensor] = []

        for step in range(inputs.shape[0]):
            keep_mask = (~reset_mask[step]).unsqueeze(-1)
            state = {
                name: value * keep_mask.to(
                    device=value.device,
                    dtype=value.dtype,
                )
                for name, value in state.items()
            }
            output, state = self._forward_with_state(
                inputs=inputs[step],
                state=state,
            )
            outputs.append(output)

        return torch.stack(outputs), state


    def get_recurrent_state(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        
        state: dict[str, torch.Tensor] = {}
        for name, module in self.named_children():
            if isinstance(module, StatefulGRU):
                state[name] = module.get_state(batch_size, device, dtype)
            elif isinstance(module, ConfigurableNetwork):
                state.update({
                    f"{name}.{nested_name}": value
                    for nested_name, value in module.get_recurrent_state(
                        batch_size, device, dtype
                    ).items()
                })
        return state


    def reset_recurrent_state(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        
        for module in self:
            if isinstance(module, StatefulGRU):
                module.reset_state(env_ids)
            elif isinstance(module, ConfigurableNetwork):
                module.reset_recurrent_state(env_ids)


    def _forward_step(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        
        self._validate_root_input_features(inputs)
        value = inputs
        outputs: dict[str, torch.Tensor] = {}
        for name, module in self.named_children():
            value = self._resolve_inputs(inputs, value, outputs, name)
            if isinstance(module, StatefulGRU):
                value = module.forward_step(value)
            else:
                value = module(value)
            outputs[name] = value
        return value


    def _forward_with_state(
        self,
        inputs: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        self._validate_root_input_features(inputs)
        value = inputs
        next_state = dict(state)
        outputs: dict[str, torch.Tensor] = {}
        for name, module in self.named_children():
            value = self._resolve_inputs(inputs, value, outputs, name)
            value, next_state = self._forward_module_with_state(
                name=name,
                module=module,
                inputs=value,
                state=next_state,
            )
            outputs[name] = value
        return value, next_state


    def _forward_module_with_state(
        self,
        name: str,
        module: torch.nn.Module,
        inputs: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        if isinstance(module, StatefulGRU):
            output, next_module_state = module.forward_with_state(
                inputs=inputs,
                state=state[name],
            )
            state[name] = next_module_state
            return output, state
        if isinstance(module, ConfigurableNetwork) and module.is_recurrent:
            return self._forward_nested_with_state(
                name=name,
                module=module,
                inputs=inputs,
                state=state,
            )
        return module(inputs), state


    @staticmethod
    def _forward_nested_with_state(
        name: str,
        module: "ConfigurableNetwork",
        inputs: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        prefix = f"{name}."
        nested_state = {
            key.removeprefix(prefix): tensor
            for key, tensor in state.items()
            if key.startswith(prefix)
        }
        output, nested_next_state = module._forward_with_state(
            inputs=inputs,
            state=nested_state,
        )
        next_state = {
            key: tensor
            for key, tensor in state.items()
            if not key.startswith(prefix)
        }
        next_state.update({
            f"{name}.{key}": tensor
            for key, tensor in nested_next_state.items()
        })
        return output, next_state


    def _resolve_inputs(
        self,
        root_input: torch.Tensor,
        previous_output: torch.Tensor,
        outputs: Mapping[str, torch.Tensor],
        module_name: str,
    ) -> torch.Tensor:

        values: list[torch.Tensor] = []
        for reference in self._input_sources[module_name]:
            source = reference.source
            if source == "$previous":
                value = previous_output
            elif source == "obs":
                value = root_input
            elif source.startswith("obs."):
                field = source.removeprefix("obs.")
                value = root_input[..., self._input_slices[field]]
            else:
                value = outputs[source]
            if reference.feature_slice is not None:
                value = value[..., reference.feature_slice]
            values.append(value)
        return values[0] if len(values) == 1 else torch.cat(values, dim=-1)


    def _validate_root_input_features(self, inputs: torch.Tensor) -> None:
        if inputs.ndim == 0 or inputs.shape[-1] != self._root_input_features:
            raise ValueError(
                "Network input must have final dimension "
                f"{self._root_input_features}, got {tuple(inputs.shape)}."
            )


    def _validate_recurrent_state(
        self,
        state: dict[str, torch.Tensor],
    ) -> None:
        
        expected_names = set(self.get_recurrent_state_names())
        if set(state) != expected_names:
            raise ValueError("Recurrent state keys do not match the network.")


    def get_recurrent_state_names(self) -> list[str]:
        names: list[str] = []
        for name, module in self.named_children():
            if isinstance(module, StatefulGRU):
                names.append(name)
            elif isinstance(module, ConfigurableNetwork):
                names.extend(
                    f"{name}.{nested_name}"
                    for nested_name in module.get_recurrent_state_names()
                )
        return names
