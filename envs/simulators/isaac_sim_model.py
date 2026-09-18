from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from app.utils.context import RuntimeContext


class IsaacSimModelConverter:
    """Convert source robot descriptions into checkpoint-local USD assets."""

    def __init__(
        self,
        context: RuntimeContext,
        *,
        ros_package_paths: Sequence[Mapping[str, str]],
        merge_fixed_joints: bool,
        allow_self_collision: bool,
        joint_stiffness: float | Mapping[str, float] | None,
        joint_damping: float | Mapping[str, float] | None,
    ) -> None:
        self.context = context
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

        importer_config = URDFImporterConfig(
            urdf_path=str(model_path),
            usd_path=str(self._asset_output_directory(model_path)),
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
        return self._checked_import_path(
            URDFImporter(importer_config).import_urdf(),
            "URDF",
        )

    def _convert_mjcf(self, model_path: Path) -> Path:
        from isaacsim.asset.importer.mjcf import (  # pyright: ignore[reportMissingImports]
            MJCFImporter,
            MJCFImporterConfig,
        )

        importer_config = MJCFImporterConfig(
            mjcf_path=str(model_path),
            usd_path=str(self._asset_output_directory(model_path)),
            import_scene=False,
            fix_base=False,
        )
        return self._checked_import_path(
            MJCFImporter(importer_config).import_mjcf(),
            "MJCF",
        )

    def _asset_output_directory(self, model_path: Path) -> Path:
        output_directory = (
            Path(self.context.save_dir)
            / "isaac_sim_assets"
            / model_path.stem
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    @staticmethod
    def _checked_import_path(path: str, source_format: str) -> Path:
        output_path = Path(path)
        if not output_path.is_file():
            raise RuntimeError(
                f"Isaac Sim {source_format} importer did not produce a USD "
                f"file at {output_path}."
            )
        return output_path
