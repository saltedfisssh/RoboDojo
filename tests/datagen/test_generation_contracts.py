import importlib
import json
import linecache
import sys
from types import SimpleNamespace

import h5py

from src.datagen import frozen
from src.datagen.state import encode_context, terminal_checks
from src.datagen.support_demo import recorded_suffix


def test_deferred_import_uses_frozen_source_with_relative_imports(tmp_path, monkeypatch):
    package = tmp_path / "frozen_probe"
    package.mkdir()
    init = package / "__init__.py"
    child = package / "child.py"
    init.write_text("from .child import VALUE\n")
    child.write_text("VALUE = 7\n")
    monkeypatch.setattr(
        frozen, "SOURCES", {"frozen_probe/__init__.py": init.read_bytes(), "frozen_probe/child.py": child.read_bytes()}
    )
    child.write_text("raise RuntimeError('source changed while worker waited')\n")
    monkeypatch.setattr(sys, "meta_path", [frozen.FrozenFinder(tmp_path), *sys.meta_path])
    try:
        assert importlib.import_module("frozen_probe").VALUE == 7
        linecache.checkcache(str(child))
        assert linecache.getline(str(child), 1) == "VALUE = 7\n"
    finally:
        sys.modules.pop("frozen_probe.child", None)
        sys.modules.pop("frozen_probe", None)
        linecache.cache.pop(str(init), None)
        linecache.cache.pop(str(child), None)


def test_support_suffix_preserves_pauses_and_excludes_executed_frames(tmp_path):
    path = tmp_path / "demo.h5"
    contexts = [
        {"next_controls": [{"arm": 1}] * 10},
        {"next_controls": []},
        {"next_controls": [{"arm": 2}] * 3},
        {"next_controls": []},
    ]
    with h5py.File(path, "w") as f:
        f.create_dataset("context", data=[json.dumps({"interaction": encode_context(c)}) for c in contexts])
        assert recorded_suffix(f, 1) == [{}] * 10 + [{"arm": 2}] * 3 + [{}] * 7
        assert recorded_suffix(f, 3) == []


def test_final_stability_includes_trigger_payload_and_omits_historical_stages():
    cover = ("covered", {})
    uncover = ("uncovered", {})
    upright = ("upright", {})
    contained = ("liquid_contained", {})
    manager = SimpleNamespace(
        check_list=[[[cover], [uncover]]], final_check_list=[[]], trigger_check_list=[[]], func_parser=SimpleNamespace()
    )
    assert terminal_checks(manager) == [uncover]
    manager.check_list = [[]]
    manager.trigger_check_list = [[([upright], [contained], "rising_edge", False)]]
    assert terminal_checks(manager) == [upright, contained]
