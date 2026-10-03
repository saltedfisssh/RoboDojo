"""Freeze Python sources at worker startup, including imports deferred until after GPU wait."""

import hashlib
import importlib.abc
import importlib.util
import io
import linecache
from pathlib import Path
import sys
import tarfile

SOURCES = {}
REVISION = None


class FrozenLoader(importlib.abc.Loader):
    def __init__(self, path, source):
        self.path, self.source = path, source

    def create_module(self, spec):
        return None

    def is_package(self, fullname):
        return self.path.name == "__init__.py"

    def exec_module(self, module):
        # Tracebacks must display the worker's frozen source, even after later edits.
        filename = str(self.path)
        linecache.cache[filename] = (len(self.source), None, self.source.decode().splitlines(True), filename)
        exec(compile(self.source, str(self.path), "exec"), module.__dict__)


class FrozenFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        relative = fullname.replace(".", "/")
        for suffix in (".py", "/__init__.py"):
            name = relative + suffix
            if name in SOURCES:
                filename = self.root / name
                return importlib.util.spec_from_file_location(
                    fullname, filename, loader=FrozenLoader(filename, SOURCES[name])
                )
        return None


def install():
    global REVISION
    if REVISION is not None:
        return REVISION
    root = Path(__file__).resolve().parents[2]
    paths = set()
    for directory in ("src/datagen", "env", "task", "utils"):
        paths.update((root / directory).rglob("*.py"))
    for path in sorted(paths):
        SOURCES[str(path.relative_to(root))] = path.read_bytes()
    digest = hashlib.sha256()
    for name, source in SOURCES.items():
        digest.update(name.encode() + b"\0" + source)
    REVISION = digest.hexdigest()
    folder = root / ".cache/datagen-source"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{REVISION}.tar.gz"
    if not destination.exists():
        # Workers can start concurrently while waiting for the GPU. Only publish complete archives.
        import os

        temporary = destination.with_suffix(f".{os.getpid()}.partial")
        with tarfile.open(temporary, "w:gz") as archive:
            for name, source in SOURCES.items():
                info = tarfile.TarInfo(name)
                info.size = len(source)
                archive.addfile(info, io.BytesIO(source))
        temporary.replace(destination)
    sys.meta_path.insert(0, FrozenFinder(root))
    return REVISION
