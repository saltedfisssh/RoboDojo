"""Publish restored articulation links to USD without advancing simulation.

Isaac Sim 5.1 tensor set_dof_positions/update_articulations_kinematic updates
PhysX link poses. PhysX 107.3 update_transformations only publishes its simulation
output; a settled external articulation can retain old USD link poses after a
checkpoint restore. Use the 5.1 XFormPrim USD pose API with those same live poses.
Its world-pose conversion rounds small rotations through float32 matrices and
Gf.Transform.GetRotation(), so compute local poses in float64 before writing.
The check includes robot links and externally registered SingleArticulations.
"""

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


def pose_from_usd_matrix(matrix):
    """Extract XYZ/WXYZ from a USD row-vector transform, retaining small angles."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 USD transform")
    u, scale, vh = np.linalg.svd(matrix[:3, :3].T)
    rotation = u @ vh
    if scale.min() < 1e-12 or np.linalg.det(rotation) < 0:
        raise ValueError("Singular or reflected articulation transform")
    return np.r_[matrix[3, :3], mat2quat(rotation)]


def local_pose_from_world(parent_matrix, world_pose):
    """Convert a world pose with a full precision parent inverse (USD convention)."""
    world_pose = np.asarray(world_pose, dtype=np.float64)
    pose_errors(world_pose, world_pose)
    world_matrix = np.eye(4)
    world_matrix[:3, :3] = quat2mat(world_pose[3:]).T
    world_matrix[3, :3] = world_pose[:3]
    parent_matrix = np.asarray(parent_matrix, dtype=np.float64)
    pose_from_usd_matrix(parent_matrix)
    local_matrix = np.linalg.solve(parent_matrix.T, world_matrix.T).T
    return pose_from_usd_matrix(local_matrix)


def ensure_double_orientation(xform):
    """Preserve small rotations when USD converts an orient op to a matrix.

    USD 24's quatf matrix conversion loses rotations whose float scalar
    component rounds to one. Sim 5.1 XFormPrim.set_local_poses explicitly
    supports quatd orient attributes; SetTypeName authors an override in
    the current edit layer without touching the referenced robot asset.
    """
    from pxr import Gf, Sdf

    attribute = xform.GetPrim().GetAttribute("xformOp:orient")
    if attribute.GetTypeName() != Sdf.ValueTypeNames.Quatf:
        return False
    current = attribute.Get()
    quaternion = np.r_[current.GetReal(), np.asarray(current.GetImaginary())].astype(np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid USD orientation before precision promotion")
    if not attribute.SetTypeName(Sdf.ValueTypeNames.Quatd):
        raise RuntimeError("Unable to promote USD articulation orientation to double precision")
    attribute.Set(Gf.Quatd(*(quaternion / norm)))
    return True


def pose_errors(actual, expected):
    """Position and sign-invariant unit-quaternion errors, in metres/radians."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape or actual.shape[-1] != 7:
        raise ValueError("Expected matching arrays of XYZ + WXYZ poses")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite visual or physics pose")
    a, b = actual[..., 3:], expected[..., 3:]
    na, nb = np.linalg.norm(a, axis=-1, keepdims=True), np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(na < 1e-8) or np.any(nb < 1e-8):
        raise ValueError("Invalid zero quaternion")
    a, b = a / na, b / nb
    chord = np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1))
    angle = 4 * np.arcsin(np.clip(chord / 2, 0, 1))
    return np.linalg.norm(actual[..., :3] - expected[..., :3], axis=-1), angle


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class ArticulationVisualState:
    def __init__(self, env, external):
        from isaacsim.core.prims import XFormPrim
        from pxr import UsdGeom

        self.env = env
        physics_views = [asset.root_physx_view for asset in env.sim.scene.articulations.values()]
        physics_views.extend(obj._articulation_view._physics_view for obj in external.values())
        self.sources, self.paths, self.xforms = [], [], []
        stage = env.scene_manager.layout_manager.stage if hasattr(env.scene_manager.layout_manager, "stage") else None
        if stage is None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        for physics in physics_views:
            paths = [path for group in physics.link_paths for path in group]
            if set(paths) & set(self.paths):
                raise ValueError("Duplicate articulation link in visual checkpoint schema")
            self.sources.append(physics)
            self.paths.extend(paths)
        for path in self.paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.GetAttribute("xformOp:translate") or not prim.GetAttribute("xformOp:orient"):
                raise ValueError(f"Restored link lacks PhysX translate/orient properties: {path}")
            self.xforms.append(UsdGeom.Xformable(prim))
        # Resolve parent transforms only after any parent body has been updated.
        self.groups = []
        for depth in sorted({path.count("/") for path in self.paths}):
            paths = [path for path in self.paths if path.count("/") == depth]
            view = XFormPrim(paths, reset_xform_properties=False)
            self.groups.append((view, [self.paths.index(path) for path in view.prim_paths]))

    def physics_poses(self):
        if not self.sources:
            return np.empty((0, 7))
        poses = np.concatenate([as_numpy(view.get_link_transforms()).reshape(-1, 7) for view in self.sources])
        return poses[:, [0, 1, 2, 6, 3, 4, 5]].copy()

    def usd_poses(self):
        from pxr import Usd

        poses = []
        for xform in self.xforms:
            transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            poses.append(pose_from_usd_matrix(transform))
        return np.asarray(poses).reshape(-1, 7)

    def errors(self):
        distance, angle = pose_errors(self.usd_poses(), self.physics_poses())
        return {
            "position_m": float(distance.max(initial=0)),
            "angle_rad": float(angle.max(initial=0)),
        }

    def publish(self):
        from pxr import Usd, UsdGeom
        import torch

        poses = self.physics_poses()
        distance, angle = pose_errors(self.usd_poses(), poses)
        changed = (distance > 1e-6) | (angle > 1e-5)
        for view, indices in self.groups:
            selected = [i for i, source in enumerate(indices) if changed[source]]
            if not selected:
                continue
            for index in selected:
                ensure_double_orientation(self.xforms[indices[index]])
            targets = poses[np.asarray(indices)[selected]]
            local = np.asarray(
                [
                    local_pose_from_world(
                        UsdGeom.Xformable(
                            self.xforms[indices[index]].GetPrim().GetParent()
                        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                        target,
                    )
                    for index, target in zip(selected, targets, strict=True)
                ]
            )
            view.set_local_poses(
                translations=torch.as_tensor(local[:, :3], dtype=torch.float64, device=self.env.sim.scene.device),
                orientations=torch.as_tensor(local[:, 3:], dtype=torch.float64, device=self.env.sim.scene.device),
                indices=selected,
            )
        errors = self.errors()
        if errors["position_m"] > 1e-4 or errors["angle_rad"] > 1e-4:
            actual = self.usd_poses()
            delta_position, delta_angle = pose_errors(actual, poses)
            worst = int(np.argmax(np.maximum(delta_position, delta_angle)))
            xform = self.xforms[worst]
            detail = {
                "path": self.paths[worst],
                "usd": actual[worst].tolist(),
                "physics": poses[worst].tolist(),
                "reset_xform_stack": bool(xform.GetResetXformStack()),
                "ops": [str(op.GetOpName()) for op in xform.GetOrderedXformOps()],
                "initially_selected": bool(changed[worst]),
            }
            raise RuntimeError(f"Restored articulation visuals differ from physics: {errors}; worst={detail}")
        return {
            **errors,
            "published_links": int(changed.sum()),
            "before_position_m": float(distance.max(initial=0)),
            "before_angle_rad": float(angle.max(initial=0)),
        }
