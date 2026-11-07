# pyright: reportAttributeAccessIssue=false
"""MuJoCo heightfield adapter for the shared terrain-v1 geometry.

Each model owns a fixed-size heightfield buffer. Reset changes elevations, not
model topology or bounding volumes. The original floor geom keeps its ID and
contact/material settings. Meshes are deliberately avoided: MuJoCo collides
their convex hulls, which would fill the valleys of rough terrain.
"""
from __future__ import annotations

import mujoco
import numpy as np
from pathlib import Path

from envs.simulators.terrain import TerrainGeometry, TerrainPatch


class MujocoTerrain:

    def __init__(
        self,
        geometry: TerrainGeometry,
        num_envs: int,
    ):
        
        self.geometry = geometry
        self.num_envs = num_envs
        self.grids: list[np.ndarray] = []


    def build_models(
        self,
        path: Path,
        floor_names: tuple[str, ...],
    ) -> list:

        if len(floor_names) != 1:
            raise ValueError("Geometric terrain requires exactly one named floor geom.")
        spec = mujoco.MjSpec.from_file(str(path))
        floor = spec.geom(floor_names[0])
        if floor is None:
            raise ValueError(f"Unknown terrain floor geom: {floor_names[0]}.")

        # A local patch is aligned to world x/y; ambiguous transforms are errors.
        original = spec.compile()
        self.geom_id = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_GEOM, floor_names[0])
        if (original.geom_type[self.geom_id] != mujoco.mjtGeom.mjGEOM_PLANE
                or original.geom_bodyid[self.geom_id] != 0
                or not np.allclose(original.geom_quat[self.geom_id], [1, 0, 0, 0])):
            raise ValueError("Terrain floor must be an unrotated world-body plane.")
        if not (original.geom_contype[self.geom_id] or original.geom_conaffinity[self.geom_id]):
            raise ValueError("Terrain floor must have collision enabled.")
        if np.count_nonzero(original.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) != 1:
            raise ValueError("Remove additional planes before enabling geometric terrain.")
        self.origin = original.geom_pos[self.geom_id].copy()
        if not np.allclose(floor.pos, self.origin):
            raise ValueError("Terrain floor must not use a translated parent frame.")
        self.bound = self.geometry.height_bound
        name = "__framework_terrain"
        if any(h.name == name for h in spec.hfields):
            raise ValueError(f"Reserved terrain asset name already exists: {name}.")

        nrow, ncol = self.geometry.grid_shape
        spec.add_hfield(
            name=name,
            nrow=nrow,
            ncol=ncol,
            userdata=[0.] * (nrow * ncol),
            size=[
                self.geometry.patch_size[0] / 2,
                self.geometry.patch_size[1] / 2,
                2 * self.bound, 0.1
            ],
        )
        floor.type = mujoco.mjtGeom.mjGEOM_HFIELD
        floor.hfieldname = name
        floor.pos = self.origin - np.array([0., 0., self.bound])
        # Ignore the old plane's size; dimensions belong to the heightfield.
        floor.size = [0, 0, 0]
        models = [spec.compile() for _ in range(self.num_envs)]
        self.hfield_id = int(models[0].geom_dataid[self.geom_id])
        for model in models:
            # Buffer allocation starts flat; the environment supplies geometry.
            model.hfield_data[:] = .5
            self.grids.append(np.full((nrow, ncol), self.origin[2], dtype=np.float64))
        return models


    def validate_patch(
        self,
        patch: TerrainPatch,
    ) -> None:

        heights = np.asarray(patch.heights, dtype=np.float64)
        nrow, ncol = self.geometry.grid_shape
        half_x, half_y = (length / 2 for length in self.geometry.patch_size)
        if (
            heights.shape != (nrow, ncol)
            or len(patch.x) != ncol
            or len(patch.y) != nrow
            or not np.isfinite(heights).all()
            or np.any(np.abs(heights) > self.bound + 1e-12)
            or not np.allclose(patch.x, np.linspace(-half_x, half_x, ncol), rtol=0, atol=1e-12)
            or not np.allclose(patch.y, np.linspace(-half_y, half_y, nrow), rtol=0, atol=1e-12)
        ):
            raise ValueError("Supplied terrain patch does not fit the collision geometry layout.")


    def install(
        self,
        model,
        patch: TerrainPatch,
    ) -> np.ndarray:

        heights = np.asarray(patch.heights, dtype=np.float64)
        start = model.hfield_adr[self.hfield_id]
        count = heights.size
        # Writing compiled data avoids the XML compiler's row flip and automatic
        # min/max normalization. Both the shared grid and mjModel use y ascending.
        model.hfield_data[start:start + count] = ((heights + self.bound) / (2 * self.bound)).ravel()
        return (model.hfield_data[start:start + count].astype(np.float64)
                .reshape(heights.shape) * (2 * self.bound) - self.bound + self.origin[2])


    def heights(
        self,
        env_id: int,
        xy: np.ndarray,
    ) -> np.ndarray:
        """Interpolate the exact lower-left/upper-right collision triangles."""

        grid = self.grids[env_id]
        half = np.asarray(self.geometry.patch_size) / 2
        local = xy - self.origin[:2]
        if not np.isfinite(local).all() or np.any(np.abs(local) > half + 1e-7):
            raise ValueError("Ground query lies outside the finite terrain patch or is non-finite.")
        uv = (np.clip(local, -half, half) + half) / (2 * half) * np.array([grid.shape[1] - 1, grid.shape[0] - 1])
        i = np.minimum(uv[:, 0].astype(int), grid.shape[1] - 2)
        j = np.minimum(uv[:, 1].astype(int), grid.shape[0] - 2)
        u, v = uv[:, 0] - i, uv[:, 1] - j
        a, b, c, d = grid[j, i], grid[j, i + 1], grid[j + 1, i + 1], grid[j + 1, i]
        return np.where(u >= v, a + u * (b - a) + v * (c - b),
                        a + u * (c - d) + v * (d - a))
