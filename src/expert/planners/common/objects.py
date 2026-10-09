"""Object-specific grasp/placement geometry, independent of task sequencing."""

from abc import ABC, abstractmethod

import numpy as np

from src.expert.planner import wrap


def arm_order(motion, xy):
    first = motion.arm_for(xy)
    return first, 1 - first


def box_yaws(yaw, step=np.pi / 2):
    return [wrap(yaw + k * step) for k in range(int(round(2 * np.pi / step)))]


class ObjectProfile(ABC):
    """Geometry and manipulation parameters for a family of rigid objects."""

    noun = "object"
    lift = 0.08
    place_options = {}

    def __init__(self, motion):
        self.motion = motion
        self.rest_height = {
            label: motion.pose(label)[0][2] - motion.bottom_z(label) for label in motion.labels("Rigid")
        }

    @abstractmethod
    def grasp(self, label):
        """Return (TCP position or yaw-dependent position function, jaw yaws)."""


class StackProfile(ObjectProfile):
    max_carry = 0.45

    @abstractmethod
    def stack_target(self, label, support_xy, support_top, level):
        """Object-origin target for a particular support height and stack level."""

    def added_height(self, label):
        return float(np.ptp(self.motion.world_vertices(label)[:, 2]))
