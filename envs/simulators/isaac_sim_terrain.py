"""Build and query the shared static Isaac terrain surface."""

import math
import torch
import numpy as np
from collections.abc import Sequence
from itertools import product
from typing import Any

from .terrain import (
    TerrainCatalog, TerrainGeometry, TerrainPatch, TerrainRegionKey, TerrainSpawnGeometry,
)


class IsaacSimTerrain:

    prim_path = "/World/TerrainFloor"
    max_triangles = 2_000_000

    def __init__(
        self,
        geometry: TerrainGeometry,
        num_envs: int,
        seed: int,
        layout: tuple[int, tuple[int, int], float],
        device: torch.device,
    ) -> None:

        self.geometry = geometry
        self.layout = layout
        variants, _, self.border_width = layout
        catalog = geometry.catalog
        if catalog is None or not catalog.conditions:
            raise ValueError(
                "Isaac shared terrain requires a non-empty patch catalog."
            )

        conditions = catalog.conditions
        region_cols, x_cells, y_cells = self._map_dimensions(
            len(conditions) * variants
        )
        keys = [
            (category, difficulty, variant)
            for category, difficulty in conditions
            for variant in range(variants)
        ]
        self._region_ids = {key: index for index, key in enumerate(keys)}

        size_x, size_y = geometry.patch_size
        rows, cols = geometry.grid_shape
        dx, dy = size_x / (cols - 1), size_y / (rows - 1)
        self._half_size = torch.tensor(
            [size_x / 2, size_y / 2], dtype=torch.float64, device=device
        )
        self._cell_size = self._half_size.new_tensor([dx, dy])
        self._map_lower = self._half_size.new_tensor(
            [-x_cells * dx / 2, -y_cells * dy / 2]
        )
        self._map_upper = -self._map_lower

        grid = np.zeros((y_cells + 1, x_cells + 1), dtype=np.float32)
        self._patches, origins = self._populate_grid(
            grid, catalog, seed, keys, region_cols, (dx, dy)
        )
        self._world_grid = torch.as_tensor(grid, device=device)
        self.origins = self._half_size.new_tensor(origins)
        self.region_ids = torch.zeros(num_envs, dtype=torch.long, device=device)


    def _map_dimensions(
        self,
        num_regions: int,
    ) -> tuple[int, int, int]:
        """Return the region column count and total map cells along x and y."""

        _, (tiles_x, tiles_y), _ = self.layout
        geometry = self.geometry
        size_x, size_y = geometry.patch_size
        rows, cols = geometry.grid_shape
        region_x_cells = tiles_x * (cols - 1)
        region_y_cells = tiles_y * (rows - 1)
        if 2 * num_regions * region_x_cells * region_y_cells > self.max_triangles:
            raise ValueError(
                f"Isaac terrain exceeds the {self.max_triangles:,}-triangle budget. "
                "Increase terrain resolution in metres or reduce the region size or variant count."
            )
        aspect = tiles_y * size_y / (tiles_x * size_x)
        region_cols = min(
            num_regions,
            max(1, math.ceil(math.sqrt(num_regions * aspect)))
        )
        region_rows = math.ceil(num_regions / region_cols)
        x_cells = region_cols * region_x_cells
        y_cells = region_rows * region_y_cells
        border_cells = 2 if self.border_width else 0
        triangles = 2 * (x_cells + border_cells) * (y_cells + border_cells)
        if triangles > self.max_triangles:
            raise ValueError(
                f"Isaac terrain requires {triangles:,} triangles (limit {self.max_triangles:,}). "
                "Increase terrain resolution in metres or reduce the region size or variant count."
            )
        return region_cols, x_cells, y_cells


    def _populate_grid(
        self,
        grid: np.ndarray,
        catalog: TerrainCatalog,
        seed: int,
        keys: Sequence[TerrainRegionKey],
        region_cols: int,
        cell_size: tuple[float, float],
    ) -> tuple[dict[TerrainRegionKey, TerrainPatch], list[tuple[float, float, float]]]:
        """Lay out the generated patches and locate each region's spawn tile."""

        _, (tiles_x, tiles_y), _ = self.layout
        size_x, size_y = self.geometry.patch_size
        rows, cols = self.geometry.grid_shape
        dx, dy = cell_size
        x_cells, y_cells = grid.shape[1] - 1, grid.shape[0] - 1
        local_x = np.linspace(-size_x / 2, size_x / 2, cols)
        local_y = np.linspace(-size_y / 2, size_y / 2, rows)
        origins = []
        patches = {}
        for index, key in enumerate(keys):
            row, col = divmod(index, region_cols)
            x_start = col * tiles_x * (cols - 1)
            y_start = row * tiles_y * (rows - 1)
            patch = catalog.patch(seed, key)
            heights = self._validated_grid(patch, local_x, local_y)
            for tile_y in range(tiles_y):
                y = y_start + tile_y * (rows - 1)
                for tile_x in range(tiles_x):
                    x = x_start + tile_x * (cols - 1)
                    grid[y:y + rows, x:x + cols] = heights
            patches[key] = patch
            origins.append((
                -x_cells * dx / 2 + (x_start + (tiles_x // 2 + .5) * (cols - 1)) * dx,
                -y_cells * dy / 2 + (y_start + (tiles_y // 2 + .5) * (rows - 1)) * dy,
                0.,
            ))
        return patches, origins


    def _validated_grid(
        self,
        patch: TerrainPatch,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:

        heights = np.asarray(patch.heights, dtype=np.float64)
        if (
            heights.shape != self.geometry.grid_shape
            or len(patch.x) != len(x)
            or len(patch.y) != len(y)
            or not np.allclose(patch.x, x, rtol=0, atol=1e-12)
            or not np.allclose(patch.y, y, rtol=0, atol=1e-12)
            or not np.isfinite(heights).all()
            or np.any(np.abs(heights) > self.geometry.height_bound + 1e-12)
            or np.any(np.abs(heights[[0, -1], :]) > 1e-12)
            or np.any(np.abs(heights[:, [0, -1]]) > 1e-12)
        ):
            raise ValueError(
                "Shared terrain patches must match the grid and meet at zero height."
            )
        grid = heights.astype(np.float32)
        grid[[0, -1], :] = 0.
        grid[:, [0, -1]] = 0.
        return grid


    def _mesh_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        grid = self._world_grid.cpu().numpy()
        dx, dy = self._cell_size.tolist()
        rows, cols = grid.shape
        x = (np.arange(cols) - (cols - 1) / 2) * dx
        y = (np.arange(rows) - (rows - 1) / 2) * dy
        if self.border_width:
            x = np.r_[x[0] - self.border_width, x, x[-1] + self.border_width]
            y = np.r_[y[0] - self.border_width, y, y[-1] + self.border_width]
            grid = np.pad(grid, 1)
        xx, yy = np.meshgrid(x, y)
        vertices = np.column_stack(
            (xx.ravel(), yy.ravel(), grid.ravel())
        ).astype(np.float32)
        nodes = np.arange(xx.size, dtype=np.int32).reshape(xx.shape)
        a = nodes[:-1, :-1].ravel()
        b = nodes[:-1, 1:].ravel()
        c = nodes[1:, 1:].ravel()
        d = nodes[1:, :-1].ravel()
        faces = np.column_stack((a, b, c, a, c, d)).ravel()
        return vertices, faces


    def build(
        self,
        stage: Any,
    ) -> None:

        from pxr import UsdGeom, UsdPhysics, UsdShade, Vt

        material: Any = UsdShade.Material.Define(
            stage, "/World/TerrainPhysicsMaterial"
        )
        physics_material: Any = UsdPhysics.MaterialAPI.Apply(
            material.GetPrim()
        )
        physics_material.CreateStaticFrictionAttr(.5)
        physics_material.CreateDynamicFrictionAttr(.5)
        vertices, faces = self._mesh_arrays()
        mesh: Any = UsdGeom.Mesh.Define(stage, self.prim_path)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(
            np.full(len(faces) // 3, 3, dtype=np.int32)
        ))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        prim = mesh.GetPrim()
        binding: Any = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(material, materialPurpose="physics")
        mesh_collision: Any = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr("none")
        UsdPhysics.CollisionAPI.Apply(prim)


    def get_patch(
        self,
        key: TerrainRegionKey,
    ) -> TerrainPatch:
        return self._patches[key]


    def select(
        self,
        indices: torch.Tensor,
        keys: Sequence[TerrainRegionKey],
    ) -> None:
        if len(keys) != indices.numel():
            raise ValueError(
                "Terrain regions must match the selected environments."
            )
        selected_regions = [self._region_ids[key] for key in keys]
        self.region_ids[indices] = self.region_ids.new_tensor(selected_regions)


    def bounds(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centres = self.origins[self.region_ids[indices], :2]
        return centres - self._half_size, centres + self._half_size


    def heights(
        self,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        inside = ((xy >= self._map_lower) & (xy <= self._map_upper)).all(dim=-1)
        covered = (
            (xy >= self._map_lower - self.border_width)
            & (xy <= self._map_upper + self.border_width)
        ).all(dim=-1)
        rows, cols = self._world_grid.shape
        grid_xy = ((xy - self._map_lower) / self._cell_size).nan_to_num()
        grid_x = grid_xy[..., 0].clamp(0, cols - 1)
        grid_y = grid_xy[..., 1].clamp(0, rows - 1)
        i = grid_x.long().clamp_max(cols - 2)
        j = grid_y.long().clamp_max(rows - 2)
        u, v = grid_x - i, grid_y - j
        a = self._world_grid[j, i]
        b = self._world_grid[j, i + 1]
        c = self._world_grid[j + 1, i + 1]
        d = self._world_grid[j + 1, i]
        elevation = torch.where(
            u >= v,
            a + u * (b - a) + v * (c - b),
            a + u * (c - d) + v * (d - a)
        )
        return torch.where(covered, torch.where(inside, elevation, 0.), torch.nan).to(xy)


    def ground_state(
        self,
        xy: torch.Tensor,
        indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        heights = self.heights(xy)
        valid = torch.isfinite(heights).all(dim=1)
        lower, upper = self.bounds(indices)
        in_bounds = ((xy >= lower[:, None]) & (xy <= upper[:, None])).all(dim=(1, 2))
        heights = torch.where(valid[:, None], heights, 0.)
        return {
            "base_ground_height": heights[:, 0],
            "foot_ground_heights": heights[:, 1:],
            "ground_valid": valid,
            "terrain_in_bounds": in_bounds,
        }


def measure_spawn_geometry(
    stage: Any,
    robot_path: str,
    body_paths: Sequence[str],
    body_positions: np.ndarray,
    body_orientations: np.ndarray,
    origin: np.ndarray,
    root_position: np.ndarray,
) -> TerrainSpawnGeometry:
    """Bound every collision shape at the configured articulation pose."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    robot = stage.GetPrimAtPath(robot_path)
    body_ids = {robot_path + path: i for i, path in enumerate(body_paths)}
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "proxy", "render", "guide"])
    points = []
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision: Any = UsdPhysics.CollisionAPI(prim)
        if not collision.GetCollisionEnabledAttr().Get():
            continue
        body = prim
        while str(body.GetPath()) not in body_ids and body != robot:
            body = body.GetParent()
        index = body_ids.get(str(body.GetPath()))
        if index is None:
            raise ValueError(f"Robot collision shape {prim.GetPath()} has no tracked rigid body.")
        relative_bound: Any = bounds.ComputeRelativeBound(prim, body)
        box = relative_bound.ComputeAlignedRange()
        if box.IsEmpty():
            raise ValueError(f"Cannot measure robot collision shape {prim.GetPath()}.")
        w, x, y, z = (float(q) for q in body_orientations[index])
        rotation = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        corners = product(*zip(box.GetMin(), box.GetMax()))
        points.extend(
            np.array(rotation.TransformDir(Gf.Vec3d(*corner))) + body_positions[index]
            for corner in corners
        )
    if not points:
        raise ValueError("Terrain spawn measurement requires robot collision shapes.")
    return TerrainSpawnGeometry(
        tuple(origin.tolist()),
        tuple(root_position.tolist()),
        tuple((np.min(points, axis=0) - root_position).tolist()),
        tuple((np.max(points, axis=0) - root_position).tolist()),
    )
