"""Per-environment foot materials for Isaac's shared collision ground."""
from collections.abc import Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, cast


_COMBINE_MODES = ("average", "min", "multiply", "max")
_MATERIAL_ROOT = "/World/EnvironmentFrictionMaterials"
_FLOAT32_MAX = 3.4028234663852886e38


@dataclass(frozen=True)
class _Surface:

    static: float
    dynamic: float
    combine: str


def _combine(
    foot: float,
    ground: float,
    mode: str,
) -> float:

    if mode == "average":
        return (foot + ground) / 2
    if mode == "min":
        return min(foot, ground)
    if mode == "multiply":
        return foot * ground
    return max(foot, ground)


def _compensate(
    foot: _Surface,
    floors: set[_Surface],
    combine: str,
) -> tuple[float, float]:

    static_candidates = []
    dynamic_candidates = []
    for ground in floors:
        original = max(foot.combine, ground.combine, key=_COMBINE_MODES.index)
        targets = (
            _combine(foot.static, ground.static, original),
            _combine(foot.dynamic, ground.dynamic, original),
        )
        for target, coefficient, candidates in zip(
            targets, (ground.static, ground.dynamic),
            (static_candidates, dynamic_candidates), strict=True,
        ):
            if combine == "max":
                # On a zero-friction floor, max(foot, 0) is exactly foot.
                candidates.append(target)
            elif coefficient > 0:
                candidates.append(target / coefficient)
            elif target > 0:
                raise ValueError(
                    "Cannot reproduce positive contact friction on a zero-friction channel."
                )

    coefficients = []
    for candidates in (static_candidates, dynamic_candidates):
        first = min(candidates, default=0.)
        if any(not isclose(value, first, rel_tol=1e-6) for value in candidates):
            raise ValueError(
                "Isaac floors require different compensation for the same foot collider."
            )
        coefficients.append(first)
    static, dynamic = coefficients
    if max(static, dynamic) > _FLOAT32_MAX or static < dynamic:
        raise ValueError("Isaac floors cannot represent the compensated foot friction.")
    return static, dynamic


@dataclass(frozen=True)
class _FootMaterial:

    static_attr: Any
    dynamic_attr: Any
    static_target: float
    dynamic_target: float

    def coefficients(self, factor: float) -> tuple[float, float]:
        return self.static_target * factor, self.dynamic_target * factor


@dataclass(frozen=True)
class _FrictionSnapshot:

    owner: "IsaacSimFriction"
    env_ids: tuple[int, ...]
    factors: tuple[float, ...]


class IsaacSimFriction:

    def __init__(
        self,
        stage: Any,
        floor_paths: Sequence[str],
        foot_paths_by_env: Sequence[Sequence[str]],
    ) -> None:

        from pxr import PhysxSchema  # pyright: ignore[reportAttributeAccessIssue]
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        # The SDK's generated USD stubs misdeclare several schema return types.
        sdf, usd, geom, physics_schema, shade = (
            cast(Any, module) for module in (Sdf, Usd, UsdGeom, UsdPhysics, UsdShade)
        )

        def bound_material(prim: Any) -> Any:
            material, _ = shade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            return material

        def read_surface(material: Any) -> _Surface:
            if not material or not material.GetPrim().HasAPI(physics_schema.MaterialAPI):
                # PhysX's implicit rigid contact material.
                return _Surface(.5, .5, "average")
            prim = material.GetPrim()
            physics = physics_schema.MaterialAPI(prim)
            physx = PhysxSchema.PhysxMaterialAPI(prim)
            mode = physx.GetFrictionCombineModeAttr().Get() if physx else "average"
            static = physics.GetStaticFrictionAttr().Get()
            dynamic = physics.GetDynamicFrictionAttr().Get()
            if (
                mode not in _COMBINE_MODES or static is None or dynamic is None
                or not isfinite(static) or not isfinite(dynamic) or not 0 <= dynamic <= static
            ):
                raise ValueError(f"Unsupported Isaac friction material: {prim.GetPath()}.")
            return _Surface(static, dynamic, mode)

        floors = set()
        for path in floor_paths:
            prim = stage.GetPrimAtPath(path)
            if any(subset.HasAPI(shade.MaterialBindingAPI)
                   for subset in geom.Subset.GetAllGeomSubsets(geom.Imageable(prim))):
                raise ValueError("Isaac friction requires a uniform shared floor material.")
            floors.add(read_surface(bound_material(prim)))
        if not floors:
            raise ValueError("Isaac friction requires a collision floor.")
        combine = "max" if all(floor.static == 0 for floor in floors) else "multiply"
        if combine == "multiply" and any(floor.combine == "max" for floor in floors):
            raise ValueError(
                "A floor with max combine mode overrides Isaac friction compensation."
            )

        if stage.GetPrimAtPath(_MATERIAL_ROOT):
            raise ValueError(f"Reserved Isaac friction prim already exists: {_MATERIAL_ROOT}.")

        groups_by_env: list[list[tuple[str, _Surface, list[Any]]]] = []
        for body_paths in foot_paths_by_env:
            groups: dict[str, list[Any]] = {}
            for path in body_paths:
                body = stage.GetPrimAtPath(path)
                shapes = []
                descendants = iter(usd.PrimRange(body, usd.TraverseInstanceProxies()))
                for prim in descendants:
                    if prim != body and prim.HasAPI(physics_schema.RigidBodyAPI):
                        descendants.PruneChildren()
                        continue
                    if (prim.HasAPI(physics_schema.CollisionAPI)
                            and physics_schema.CollisionAPI(prim).GetCollisionEnabledAttr().Get()):
                        shapes.append(prim)
                if not shapes:
                    raise ValueError(f"Isaac friction requires a collider on foot {path}.")
                for prim in shapes:
                    if prim.IsInstanceProxy() or any(
                        subset.HasAPI(shade.MaterialBindingAPI)
                        for subset in geom.Subset.GetAllGeomSubsets(geom.Imageable(prim))
                    ):
                        raise ValueError(
                            f"Unsupported Isaac foot collider for friction: {prim.GetPath()}."
                        )
                    material = bound_material(prim)
                    material_path = str(material.GetPath()) if material else ""
                    groups.setdefault(material_path, []).append(prim)
            groups_by_env.append([
                (path, read_surface(shade.Material.Get(stage, path) if path else None), shapes)
                for path, shapes in groups.items()
            ])

        self._layer = sdf.Layer.CreateAnonymous("EnvironmentFriction.usda")
        self._session_layer = stage.GetSessionLayer()
        self._factors = [1.] * len(foot_paths_by_env)
        self._materials: list[list[_FootMaterial]] = []
        session_layers = self._session_layer.subLayerPaths
        session_layers.insert(0, self._layer.identifier)
        try:
            with usd.EditContext(stage, self._layer):
                for env_id, material_groups in enumerate(groups_by_env):
                    materials = []
                    for group, (original_path, foot, shapes) in enumerate(material_groups):
                        path = f"{_MATERIAL_ROOT}/env_{env_id}_{group}"
                        material = shade.Material.Define(stage, path)
                        prim = material.GetPrim()
                        if original_path:
                            prim.GetReferences().AddInternalReference(original_path)
                        physics = physics_schema.MaterialAPI.Apply(prim)
                        PhysxSchema.PhysxMaterialAPI.Apply(prim).CreateFrictionCombineModeAttr(combine)
                        static, dynamic = _compensate(foot, floors, combine)
                        static_attr = physics.CreateStaticFrictionAttr(static)
                        dynamic_attr = physics.CreateDynamicFrictionAttr(dynamic)
                        materials.append(_FootMaterial(
                            self._layer.GetAttributeAtPath(static_attr.GetPath()),
                            self._layer.GetAttributeAtPath(dynamic_attr.GetPath()),
                            static, dynamic,
                        ))
                        for shape in shapes:
                            binding = shade.MaterialBindingAPI.Apply(shape)
                            binding.Bind(material, materialPurpose="physics")
                            effective = bound_material(shape)
                            if not effective or effective.GetPath() != material.GetPath():
                                raise ValueError(
                                    f"Cannot override Isaac foot material on {shape.GetPath()} "
                                    f"(resolved {effective.GetPath() if effective else 'none'})."
                                )
                    self._materials.append(materials)
        except Exception:
            session_layers.remove(self._layer.identifier)
            raise


    def unbind(self) -> None:
        self._session_layer.subLayerPaths.remove(self._layer.identifier)


    def capture(
        self,
        env_ids: Sequence[int],
    ) -> object:
        return _FrictionSnapshot(self, tuple(env_ids), tuple(self._factors[i] for i in env_ids))


    def apply(
        self,
        env_ids: Sequence[int],
        factors: Sequence[float],
    ) -> None:

        from pxr import Sdf

        if (len(env_ids) != len(factors) or len(set(env_ids)) != len(env_ids)
                or any(not 0 <= env_id < len(self._factors) for env_id in env_ids)):
            raise ValueError("Invalid Isaac environment selection for friction.")
        pending = []
        for env_id, factor in zip(env_ids, factors, strict=True):
            if not isfinite(factor) or factor < 0:
                raise ValueError("Isaac friction factors must be finite and non-negative.")
            if self._factors[env_id] == factor:
                continue
            for material in self._materials[env_id]:
                static, dynamic = material.coefficients(factor)
                if not all(isfinite(value) and value <= _FLOAT32_MAX for value in (static, dynamic)):
                    raise ValueError("Compensated Isaac friction coefficients exceed float32 range.")
                pending.append((material, static, dynamic))
        if not pending:
            return
        # If a write fails, the term will restore the pre-reset snapshot.
        # Record the new factors first so rollback also rewrites partial edits.
        for env_id, factor in zip(env_ids, factors, strict=True):
            self._factors[env_id] = factor
        with Sdf.ChangeBlock():
            for material, static, dynamic in pending:
                material.static_attr.default = static
                material.dynamic_attr.default = dynamic


    def restore(
        self,
        snapshot: object,
    ) -> None:
        if not isinstance(snapshot, _FrictionSnapshot) or snapshot.owner is not self:
            raise TypeError("Invalid Isaac friction snapshot.")
        self.apply(snapshot.env_ids, snapshot.factors)
