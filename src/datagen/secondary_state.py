"""Explicit USD particle buffers and constant-speed conveyor checkpoints for Sim 5.1.

CPU + use_fabric=False publishes PhysX particles to USD. Buffers below are in
their authored local frame, not mislabeled as world-space cloth positions.
Only the shipped constant-speed conveyor graph is admitted; arbitrary dynamic
assets and emitters need their own state adapters.
"""

import numpy as np


class SecondaryState:
    def __init__(self, env):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Usd, UsdGeom, UsdPhysics

        self.particles, self.bodies, self.conveyors = {}, {}, {}
        self.cloth_xforms = {}
        self.kinematic_bodies = set()
        lm = env.scene_manager.layout_manager
        for kind in ("Garment", "Fluid", "Dynamic"):
            for record in lm.get_layout_records(0, kind):
                label = record["label"]
                obj = lm.get_scene_object(0, record["inst_name"])
                if kind in ("Garment", "Fluid"):
                    prim = obj.prim if kind == "Garment" else obj.point_instancer.GetPrim()
                    geometry = UsdGeom.Mesh(prim) if kind == "Garment" else UsdGeom.PointInstancer(prim)
                    positions = geometry.GetPointsAttr() if kind == "Garment" else geometry.GetPositionsAttr()
                    velocities = geometry.GetVelocitiesAttr()
                    if velocities.Get() is None:
                        raise ValueError(f"{kind} {label} has no live velocity buffer")
                    self.particles[label] = (kind, positions, velocities)
                    if kind == "Garment":
                        from src.datagen.particle_transforms import cloth_xform_ops

                        self.cloth_xforms[label] = cloth_xform_ops(prim)
                else:
                    if record["category"] != "conveyor" or record["category_idx"] != 0:
                        raise ValueError("Only the shipped constant-speed conveyor has a dynamic state adapter")
                    root = obj.stage.GetPrimAtPath(obj.prim_path)
                    attributes = []
                    allowed = {
                        "omni.graph.action.OnPlaybackTick",
                        "isaacsim.asset.gen.conveyor.IsaacConveyor",
                        "omni.graph.core.ReadVariable",
                    }
                    for prim in Usd.PrimRange(root):
                        node_type = prim.GetAttribute("node:type").Get()
                        if node_type and node_type not in allowed:
                            raise ValueError(f"Conveyor graph has an unsupported stateful node: {node_type}")
                        for attr in prim.GetAttributes():
                            name = attr.GetName()
                            if name.startswith("physxSurfaceVelocity:") or name == "graph:variable:Velocity":
                                attributes.append(attr)
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            identity = f"{label}/{prim.GetPath().MakeRelativePath(root.GetPath())}"
                            body = SingleRigidPrim(str(prim.GetPath()), reset_xform_properties=False)
                            body.initialize()
                            self.bodies[identity] = body
                            if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
                                self.kinematic_bodies.add(identity)
                    self.conveyors[label] = attributes

    @property
    def enabled(self):
        return bool(self.particles or self.conveyors)

    def diagnostics(self):
        """Expose USD writeback and simulation-vs-visual particle buffers."""
        import carb
        from pxr import PhysxSchema

        settings = carb.settings.get_settings()
        report = {
            "usd_writeback": {
                name: settings.get(f"/physics/{name}")
                for name in ("updateToUsd", "updateParticlesToUsd", "updateVelocitiesToUsd")
            },
            "particles": {},
        }
        for label, (kind, positions, velocities) in self.particles.items():
            pos = np.asarray(positions.Get(), dtype=np.float32)
            vel = np.asarray(velocities.Get(), dtype=np.float32)
            record = {"kind": kind, "count": len(pos), "max_speed": float(np.linalg.norm(vel, axis=1).max())}
            prim = positions.GetPrim()
            if prim.HasAPI(PhysxSchema.PhysxParticleSetAPI):
                # Sim 5.1 smoothing may publish visual points separately from
                # the real solver positions. Never infer equivalence from a
                # successful readback of a freshly authored USD value.
                attr = PhysxSchema.PhysxParticleSetAPI(prim).GetSimulationPointsAttr()
                value = attr.Get()
                record["has_simulation_points"] = value is not None and len(value) > 0
                if record["has_simulation_points"]:
                    sim_pos = np.asarray(value, dtype=np.float32)
                    record["simulation_points_count"] = len(sim_pos)
                    if sim_pos.shape == pos.shape:
                        record["visual_to_simulation_max_distance"] = float(np.linalg.norm(pos - sim_pos, axis=1).max())
            report["particles"][label] = record
        return report

    def capture(self):
        arrays = {}
        for label, (_, positions, velocities) in self.particles.items():
            for field, attr in (("local_positions", positions), ("local_velocities", velocities)):
                array = np.asarray(attr.Get(), dtype=np.float32)
                if array.ndim != 2 or array.shape[1] != 3 or not len(array):
                    raise ValueError(f"Missing live particle {field} for {label}")
                arrays[f"particles/{label}/{field}"] = array
        for label, ops in self.cloth_xforms.items():
            from src.datagen.particle_transforms import capture_xform_ops, world_particle_positions

            arrays.update({f"particles/{label}/xform/{key}": value for key, value in capture_xform_ops(ops).items()})
            arrays[f"particles/{label}/world_positions"] = world_particle_positions(
                self.particles[label][1].GetPrim(), arrays[f"particles/{label}/local_positions"]
            )
        for label, body in self.bodies.items():
            pos, quat = body.get_world_pose()
            arrays[f"dynamic/{label}/pose"] = np.r_[pos, quat]
            arrays[f"dynamic/{label}/velocity"] = np.r_[body.get_linear_velocity(), body.get_angular_velocity()]
        settings = {}
        for label, attrs in self.conveyors.items():
            settings[label] = []
            for attr in attrs:
                value = attr.Get()
                settings[label].append(
                    {"name": str(attr.GetPath()), "value": list(value) if hasattr(value, "__len__") else value}
                )
        return arrays, {
            "particle_kinds": {k: v[0] for k, v in self.particles.items()},
            "cloth_xform_ops": {label: [op.GetOpName() for op in ops] for label, ops in self.cloth_xforms.items()},
            "conveyor_settings": settings,
        }

    def restore(self, arrays, context):
        from pxr import Vt

        from src.datagen.articulations import backend_like

        if context["particle_kinds"] != {k: v[0] for k, v in self.particles.items()}:
            raise ValueError("Particle type/schema differs from the checkpoint")
        for label, ops in self.cloth_xforms.items():
            from src.datagen.particle_transforms import restore_xform_ops

            if context.get("cloth_xform_ops", {}).get(label) != [op.GetOpName() for op in ops]:
                raise ValueError("Cloth transform schema changed or is missing from this legacy checkpoint")
            prefix = f"particles/{label}/xform/"
            restore_xform_ops(
                ops, {key.removeprefix(prefix): value for key, value in arrays.items() if key.startswith(prefix)}
            )
        for label, (_, positions, velocities) in self.particles.items():
            for field, attr in (("local_positions", positions), ("local_velocities", velocities)):
                attr.Set(Vt.Vec3fArray.FromNumpy(np.asarray(arrays[f"particles/{label}/{field}"], np.float32)))
        for label, body in self.bodies.items():
            pose, velocity = arrays[f"dynamic/{label}/pose"], arrays[f"dynamic/{label}/velocity"]
            body.set_world_pose(position=pose[:3], orientation=pose[3:])
            if label in self.kinematic_bodies:
                # The admitted conveyor's chassis is stationary; its belt uses
                # PhysxSurfaceVelocityAPI, restored below. PhysX rejects rigid
                # velocity setters on a kinematic actor. Moving kinematic actors
                # require a target/history adapter and are not covered here.
                if not np.allclose(velocity, 0, atol=1e-7):
                    raise ValueError(f"Moving kinematic conveyor body is unsupported: {label}")
                continue
            body.set_linear_velocity(backend_like(velocity[:3], body.get_linear_velocity()))
            body.set_angular_velocity(backend_like(velocity[3:], body.get_angular_velocity()))
        for label, attrs in self.conveyors.items():
            saved = context["conveyor_settings"][label]
            if len(attrs) != len(saved):
                raise ValueError("Conveyor settings schema changed")
            for attr, entry in zip(attrs, saved, strict=True):
                # Instance IDs can change after reconstruction; property order is stable within this hashed asset.
                if attr.GetName() != entry["name"].split(".")[-1]:
                    raise ValueError("Conveyor setting order changed")
                current = attr.Get()
                attr.Set(type(current)(*entry["value"]) if isinstance(entry["value"], list) else entry["value"])
