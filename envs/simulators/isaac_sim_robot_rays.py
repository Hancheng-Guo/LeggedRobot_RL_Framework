"""Ray queries against rigid, visible USD surfaces in their link frames."""

import numpy as np
import torch
import warp as wp
from typing import Any
from numpy.typing import NDArray
from pxr import Usd, UsdGeom


# Warp consumes these annotations when compiling the kernel body for CPU or CUDA.
@wp.kernel
def _intersect_link(
    mesh: wp.uint64,
    body: int,
    groups: int,
    points: int,
    origins: wp.array(dtype=wp.vec3),  # pyright: ignore[reportInvalidTypeForm]
    directions: wp.array(dtype=wp.vec3),  # pyright: ignore[reportInvalidTypeForm]
    poses: wp.array2d(dtype=wp.vec3),  # pyright: ignore[reportInvalidTypeForm]
    rotations: wp.array2d(dtype=wp.quat),  # pyright: ignore[reportInvalidTypeForm]
    limit: float,
    distances: wp.array(dtype=float),  # pyright: ignore[reportInvalidTypeForm]
) -> None:

    ray = wp.tid()
    group = ray // points  # pyright: ignore[reportOperatorIssue]  # 1-D launch
    env = group // groups
    direction = directions[ray]
    if wp.length_sq(direction) > 0.0:
        rotation = rotations[env, body]
        start = wp.quat_rotate_inv(rotation, origins[group] - poses[env, body])
        local_direction = wp.quat_rotate_inv(rotation, direction)
        hit = wp.mesh_query_ray(mesh, start, local_direction, limit + 1.e-6)  # pyright: ignore[reportCallIssue]
        if hit.result and hit.t <= limit:
            distances[ray] = wp.min(distances[ray], hit.t)


def _triangles(
    prim: Usd.Prim,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.integer[Any]]]:

    if prim.IsA(UsdGeom.Mesh):
        mesh: Any = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        if np.all(counts == 3):
            return vertices, indices.reshape(-1, 3)
        faces = []
        for face in np.split(indices, np.cumsum(counts)[:-1]):
            faces.extend((face[0], a, b) for a, b in zip(face[1:-1], face[2:]))
        return vertices, np.asarray(faces, dtype=indices.dtype)

    import trimesh

    if prim.IsA(UsdGeom.Cube):
        cube: Any = UsdGeom.Cube(prim)
        surface = trimesh.creation.box(extents=[cube.GetSizeAttr().Get()] * 3)
    elif prim.IsA(UsdGeom.Sphere):
        sphere: Any = UsdGeom.Sphere(prim)
        surface = trimesh.creation.icosphere(radius=sphere.GetRadiusAttr().Get())
    elif prim.IsA(UsdGeom.Cylinder):
        cylinder: Any = UsdGeom.Cylinder(prim)
        surface = trimesh.creation.cylinder(radius=cylinder.GetRadiusAttr().Get(),
                                            height=cylinder.GetHeightAttr().Get())
        axis = cylinder.GetAxisAttr().Get()
        surface.vertices = np.roll(surface.vertices, {"X": 1, "Y": 2, "Z": 0}[axis], axis=1)
    else:
        raise ValueError(f"Unsupported visible scan geometry at {prim.GetPath()}.")
    return np.asarray(surface.vertices), np.asarray(surface.faces)


class IsaacRobotRayQuery:

    def __init__(
        self,
        stage: Usd.Stage,
        root: str,
        paths: tuple[str, ...],
        device: torch.device,
    ) -> None:

        cache: Any = UsdGeom.XformCache()
        bodies = {root + path: index for index, path in enumerate(paths)}
        parts = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            visual = UsdGeom.Imageable(prim)
            if (
                visual.ComputePurpose() not in ("default", "render")
                or visual.ComputeVisibility() == "invisible"
            ):
                continue
            parent = prim
            while parent and str(parent.GetPath()) not in bodies:
                parent = parent.GetParent()
            if not parent:
                raise ValueError(f"Visible robot geometry has no rigid parent: {prim.GetPath()}")
            vertices, faces = _triangles(prim)
            transform, _ = cache.ComputeRelativeTransform(prim, parent)
            matrix = np.asarray(transform)
            local = vertices @ matrix[:3, :3] + matrix[3, :3]
            parts.setdefault(bodies[str(parent.GetPath())], []).append((local, faces))

        self.meshes: dict[int, wp.Mesh] = {}
        for body, pieces in parts.items():
            vertices, indices = [], []
            offset = 0
            for points, faces in pieces:
                vertices.append(points)
                indices.append(faces + offset)
                offset += len(points)
            self.meshes[body] = wp.Mesh(
                points=wp.array(np.concatenate(vertices), dtype=wp.vec3, device=str(device)),
                indices=wp.array(np.concatenate(indices).ravel(), dtype=wp.int32, device=str(device)),
            )

        wp.synchronize_device(str(device))


    def __call__(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        limit: float,
        positions: torch.Tensor,
        orientations: torch.Tensor,
    ) -> torch.Tensor:

        result = torch.full(directions.shape[:-1], torch.inf, device=origins.device, dtype=torch.float32)
        starts = wp.from_torch(origins.float().reshape(-1, 3).contiguous(), dtype=wp.vec3)
        vectors = wp.from_torch(directions.float().reshape(-1, 3).contiguous(), dtype=wp.vec3)
        poses = wp.from_torch(positions.float().contiguous(), dtype=wp.vec3)
        rotations = wp.from_torch(orientations[..., [1, 2, 3, 0]].float().contiguous(), dtype=wp.quat)
        distances = wp.from_torch(result.view(-1))
        stream = wp.stream_from_torch(torch.cuda.current_stream(origins.device)) if origins.is_cuda else None
        for body, mesh in self.meshes.items():
            wp.launch(
                _intersect_link, dim=result.numel(), device=str(origins.device), stream=stream,
                inputs=[
                    mesh.id, body, origins.size(1), directions.size(2), starts, vectors,
                    poses, rotations, limit, distances
                ]
            )
        return torch.where(result.isfinite(), result, torch.nan).to(origins.dtype)
