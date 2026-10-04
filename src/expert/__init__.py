"""RoboTwin-style expert data generation for RoboDojo.

Layouts are sampled from the official task YAML distribution, a scripted
expert drives the robots through the same control path as evaluation, and
only episodes that pass the official reward are kept.
"""
