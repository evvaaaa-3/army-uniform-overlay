import math
import numpy as np


def to_np(point):
    return np.array([point["x"], point["y"]], dtype=np.float32)


def midpoint(a, b):
    return (a + b) / 2.0


def distance(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def expand_segment(p1, p2, factor):
    """
    Expands a segment from its midpoint.
    factor = 0.10 means 10 percent wider on both sides.
    """
    center = midpoint(p1, p2)
    v1 = p1 - center
    v2 = p2 - center

    return center + v1 * (1 + factor), center + v2 * (1 + factor)