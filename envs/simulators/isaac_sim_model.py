from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast


class IsaacSimModelConverter:
    """Convert robot descriptions into reusable project asset files."""

    CONVERSION_FORMAT_VERSION = 1
    VIRTUAL_BODY_MASS = 1e-3
    VIRTUAL_BODY_DIAGONAL_INERTIA = (1e-6, 1e-6, 1e-6)

    def __init__(
        self,
        *,
        ros_package_paths: Sequence[Mapping[str, str]],
        merge_fixed_joints: bool,
        allow_self_collision: bool,
        joint_stiffness: float | Mapping[str, float] | None,
        joint_damping: float | Mapping[str, float] | None,
    ) -> None:
        self.ros_package_paths = tuple(ros_package_paths)
        self.merge_fixed_joints = merge_fixed_joints
        self.allow_self_collision = allow_self_collision
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping

    def convert_if_needed(self, model_path: Path) -> Path:
        suffix = model_path.suffix.lower()
        if suffix == ".urdf":
            return self._convert_urdf(model_path)
        if suffix == ".xml":
            return self._convert_mjcf(model_path)
        return model_path

    def _convert_urdf(self, model_path: Path) -> Path:
        from isaacsim.asset.importer.urdf import (  # pyright: ignore[reportMissingImports]
            URDFImporter,
            URDFImporterConfig,
        )

        output_directory = self._asset_output_directory(model_path)
        fingerprint = self._conversion_fingerprint(model_path, "urdf")
        cached_path = self._cached_import_path(output_directory, fingerprint)
        if cached_path is not None:
            return cached_path

        importer_config = URDFImporterConfig(
            urdf_path=str(model_path),
            usd_path=str(output_directory),
            merge_fixed_joints=self.merge_fixed_joints,
            allow_self_collision=self.allow_self_collision,
            ros_package_paths=list(self.ros_package_paths),
            robot_type="Quadruped",
            fix_base=False,
            joint_drive_type="force",
            joint_target_type="position",
            override_joint_stiffness=self.joint_stiffness,
            override_joint_damping=self.joint_damping,
        )
        output_path = self._checked_import_path(
            URDFImporter(importer_config).import_urdf(),
            "URDF",
        )
        self._repair_virtual_body_mass(output_path)
        self._write_conversion_manifest(
            output_directory,
            output_path,
            fingerprint,
        )
        return output_path

    def _convert_mjcf(self, model_path: Path) -> Path:
        from isaacsim.asset.importer.mjcf import (  # pyright: ignore[reportMissingImports]
            MJCFImporter,
            MJCFImporterConfig,
        )

        output_directory = self._asset_output_directory(model_path)
        fingerprint = self._conversion_fingerprint(model_path, "mjcf")
        cached_path = self._cached_import_path(output_directory, fingerprint)
        if cached_path is not None:
            return cached_path

        importer_config = MJCFImporterConfig(
            mjcf_path=str(model_path),
            usd_path=str(output_directory),
            import_scene=False,
            fix_base=False,
        )
        output_path = self._checked_import_path(
            MJCFImporter(importer_config).import_mjcf(),
            "MJCF",
        )
        self._repair_virtual_body_mass(output_path)
        self._write_conversion_manifest(
            output_directory,
            output_path,
            fingerprint,
        )
        return output_path

    def _asset_output_directory(self, model_path: Path) -> Path:
        resolved_path = model_path.resolve()
        assets_root = next(
            (
                parent
                for parent in resolved_path.parents
                if parent.name.casefold() == "assets"
            ),
            None,
        )
        if assets_root is None:
            output_directory = resolved_path.parent / model_path.stem / "USD"
        else:
            relative_path = resolved_path.relative_to(assets_root)
            model_directory_name = (
                relative_path.parts[0]
                if len(relative_path.parts) > 1
                else model_path.stem
            )
            output_directory = assets_root / model_directory_name / "USD"
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    def _conversion_fingerprint(
        self,
        model_path: Path,
        source_format: str,
    ) -> str:
        configuration = {
            "conversion_format_version": self.CONVERSION_FORMAT_VERSION,
            "source_format": source_format,
            "source_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "ros_package_names": [
                sorted(paths)
                for paths in self.ros_package_paths
            ],
            "merge_fixed_joints": self.merge_fixed_joints,
            "allow_self_collision": self.allow_self_collision,
            "joint_stiffness": self.joint_stiffness,
            "joint_damping": self.joint_damping,
        }
        payload = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _manifest_path(output_directory: Path) -> Path:
        return output_directory / "conversion.json"

    def _cached_import_path(
        self,
        output_directory: Path,
        fingerprint: str,
    ) -> Path | None:
        manifest_path = self._manifest_path(output_directory)
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if manifest.get("fingerprint") != fingerprint:
            return None
        relative_output_path = manifest.get("output_path")
        if not isinstance(relative_output_path, str):
            return None
        output_path = output_directory / relative_output_path
        return output_path if output_path.is_file() else None

    def _write_conversion_manifest(
        self,
        output_directory: Path,
        output_path: Path,
        fingerprint: str,
    ) -> None:
        relative_output_path = output_path.resolve().relative_to(
            output_directory.resolve()
        )
        manifest = {
            "fingerprint": fingerprint,
            "output_path": relative_output_path.as_posix(),
        }
        self._manifest_path(output_directory).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _repair_virtual_body_mass(self, output_path: Path) -> None:
        from pxr import Gf, Usd, UsdPhysics  # pyright: ignore[reportMissingImports]

        stage = Usd.Stage.Open(str(output_path))
        if stage is None:
            raise RuntimeError(f"Failed to open converted USD at {output_path}.")
        default_prim = stage.GetDefaultPrim()
        if default_prim.IsValid():
            variant_sets = default_prim.GetVariantSets()
            if variant_sets.HasVariantSet("Physics"):
                physics_variant = variant_sets.GetVariantSet("Physics")
                if physics_variant is None:
                    raise RuntimeError(
                        "USD Physics variant set could not be accessed."
                    )
                raw_variant_names: Any = physics_variant.GetVariantNames()
                variant_names: tuple[str, ...] = (
                    tuple(raw_variant_names)
                    if raw_variant_names is not None
                    else ()
                )
                physx_variant = next(
                    (
                        name
                        for name in variant_names
                        if name.casefold() == "physx"
                    ),
                    None,
                )
                if physx_variant is not None:
                    physics_variant.SetVariantSelection(physx_variant)

        rigid_bodies_with_colliders: set[str] = set()
        stage_prims = tuple(cast(Any, stage.Traverse()))
        for prim in stage_prims:
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            parent = prim
            while parent.IsValid() and parent != stage.GetPseudoRoot():
                if parent.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_bodies_with_colliders.add(str(parent.GetPath()))
                    break
                parent = parent.GetParent()

        modified = False
        for prim in stage_prims:
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            mass_api = (
                UsdPhysics.MassAPI(prim)
                if prim.HasAPI(UsdPhysics.MassAPI)
                else UsdPhysics.MassAPI.Apply(prim)
            )
            if mass_api is None:
                raise RuntimeError(
                    f"Failed to access mass properties for {prim.GetPath()}."
                )
            mass_attr = mass_api.GetMassAttr()
            if mass_attr is None:
                raise RuntimeError(
                    f"Failed to access mass attribute for {prim.GetPath()}."
                )
            mass = mass_attr.Get()
            if mass is not None and mass > 0.0:
                continue
            if str(prim.GetPath()) in rigid_bodies_with_colliders:
                continue
            mass_attr = mass_api.CreateMassAttr()
            inertia_attr = mass_api.CreateDiagonalInertiaAttr()
            if mass_attr is None or inertia_attr is None:
                raise RuntimeError(
                    f"Failed to create mass properties for {prim.GetPath()}."
                )
            mass_attr.Set(self.VIRTUAL_BODY_MASS)
            inertia_attr.Set(
                Gf.Vec3f(*self.VIRTUAL_BODY_DIAGONAL_INERTIA)
            )
            modified = True
        if modified:
            stage.GetRootLayer().Save()

    @staticmethod
    def _checked_import_path(path: str, source_format: str) -> Path:
        output_path = Path(path)
        if not output_path.is_file():
            raise RuntimeError(
                f"Isaac Sim {source_format} importer did not produce a USD "
                f"file at {output_path}."
            )
        return output_path
