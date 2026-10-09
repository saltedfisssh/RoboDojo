"""Episode task planners. Register entrypoints in src.expert.registry.

Keep this package lazy: importing it must not start a simulator or import all
task implementations. Shared manipulation algorithms live in ``common``.
"""
