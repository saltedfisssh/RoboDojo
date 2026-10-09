"""Exercise camera ownership without loading Isaac or a CUDA context."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def capture(monkeypatch):
    def module(name, **members):
        value = ModuleType(name)
        value.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, value)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child, value, raising=False)
        return value

    class XFormPrim:
        def __init__(self, prim_paths_expr, **kwargs):
            self.prims = list(prim_paths_expr)
            self.prim_paths = list(prim_paths_expr)
            self.name = kwargs["name"]

        def __del__(self):
            self.destroy()

        def destroy(self):
            self.callbacks_removed = True

    class Product:
        def __init__(self, **kwargs):
            self.path = f"/render/{len(products)}"
            self.destroyed = 0
            products.append(self)

        def destroy(self):
            self.destroyed += 1

    class Annotator:
        def __init__(self, name, device, **kwargs):
            self.device = device
            self.attached = []
            self.detached = []
            annotators.append(self)

        def attach(self, path):
            self.attached.append(path)

        def detach(self, paths):
            self.detached.extend(paths)

    products, annotators = [], []
    for name in ("isaacsim", "isaacsim.core", "isaacsim.sensors", "omni", "omni.replicator"):
        module(name)
    module("isaacsim.core.prims", XFormPrim=XFormPrim)
    module("isaacsim.sensors.camera", Camera=object)
    module(
        "omni.replicator.core",
        create=SimpleNamespace(render_product_tiled=Product),
        AnnotatorRegistry=SimpleNamespace(get_annotator=Annotator),
    )
    module("omni.replicator.core.scripts")
    module("omni.replicator.core.scripts.annotators", Annotator=Annotator)
    module(
        "warp",
        kernel=lambda fn: fn,
        overload=lambda *args: None,
        array=type("Array", (), {"__init__": lambda self, **kwargs: None}),
        uint8=int,
        float32=float,
        uint32=int,
        zeros=lambda shape, dtype, device: SimpleNamespace(shape=shape, device=device),
    )
    module("env.camera_manager.camera_manager", CameraManager=object)
    module("env.environment.isaac.isaac_rl_env", IsaacRLEnv=object)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        spec.loader.exec_module(loaded)
        return loaded

    load("env.camera_manager.capture.camera_view", "env/camera_manager/capture/camera_view.py")
    manager_module = load("capture_manager_test", "env/camera_manager/capture/tiled_capture_manager.py")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def cameras():
        return [[SimpleNamespace(prim_path=f"/camera/{i}", _resolution=(640, 480)) for i in range(3)]]

    owner = SimpleNamespace(cameras=cameras(), camera_names=[["head", "left", "right"]])
    config = OmegaConf.create({"annotator": {"common": {"enabled": True, "color": {"type": "rgb"}}}})
    manager = manager_module.TiledCaptureManager(1, config, owner, "cpu")
    manager.initialize(SimpleNamespace())
    return SimpleNamespace(manager=manager, owner=owner, cameras=cameras, products=products, annotators=annotators)


def test_soft_resets_reuse_three_render_products_on_selected_gpu(capture):
    for _ in range(8):
        capture.manager.reset()
    assert len(capture.products) == 3
    assert all(product.destroyed == 0 for product in capture.products)
    assert len(capture.annotators) == 3
    assert all(annotator.device == "cuda:3" for annotator in capture.annotators)
    assert all(buffers["rgb"].device == "cuda:3" for buffers in capture.manager._output_buffers.values())
    capture.manager.destroy()


def test_camera_replacement_releases_old_resources_and_destroy_is_idempotent(capture):
    manager = capture.manager
    manager.reset()
    capture.owner.cameras = capture.cameras()
    manager.reset()
    assert len(capture.products) == 6
    assert [product.destroyed for product in capture.products] == [1, 1, 1, 0, 0, 0]
    assert all(annotator.detached == annotator.attached for annotator in capture.annotators[:3])
    views = list(manager.tiled_cameras)
    manager.destroy()
    manager.destroy()
    assert all(product.destroyed == 1 for product in capture.products)
    assert all(annotator.detached == annotator.attached for annotator in capture.annotators)
    assert len(capture.owner.cameras[0]) == 3
    assert capture.owner.camera_names == [["head", "left", "right"]]
    assert not manager._output_buffers
    assert all(view.callbacks_removed for view in views)


def test_deleted_isaac_graph_node_does_not_prevent_render_product_release(capture):
    capture.manager.reset()

    def stale_graph(paths):
        raise TypeError("Invalid NodeObj object in Py_Node in getAttributes")

    capture.annotators[0].detach = stale_graph
    capture.manager.destroy()
    assert all(product.destroyed == 1 for product in capture.products)
