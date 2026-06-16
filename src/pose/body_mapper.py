import numpy as np
from src.utils.geometry import to_np, midpoint, expand_segment


class BodyMapper:
    """
    Fixed version — handles missing ankles and better expansion.
    """

    def __init__(self, visibility_threshold=0.40):
        self.visibility_threshold = visibility_threshold

    def has_points(self, pose, names):
        if pose is None:
            return False
        for name in names:
            if name not in pose:
                return False
            if pose[name]["vis"] < self.visibility_threshold:
                return False
        return True

    def torso_quad(self, pose):
        needed = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if not self.has_points(pose, needed):
            return None
        ls = to_np(pose["left_shoulder"])
        rs = to_np(pose["right_shoulder"])
        lh = to_np(pose["left_hip"])
        rh = to_np(pose["right_hip"])
        shoulder_mid = midpoint(ls, rs)
        hip_mid = midpoint(lh, rh)
        torso_vec = hip_mid - shoulder_mid
        ls, rs = expand_segment(ls, rs, 0.18)
        lh, rh = expand_segment(lh, rh, 0.14)
        ls = ls - torso_vec * 0.12
        rs = rs - torso_vec * 0.12
        lh = lh + torso_vec * 0.10
        rh = rh + torso_vec * 0.10
        return np.array([ls, rs, rh, lh], dtype=np.float32)

    def legs_quad(self, pose):
        needed_full = ["left_hip", "right_hip", "left_ankle", "right_ankle"]
        needed_min = ["left_hip", "right_hip"]
        if not self.has_points(pose, needed_min):
            return None
        lh = to_np(pose["left_hip"])
        rh = to_np(pose["right_hip"])
        if self.has_points(pose, needed_full):
            la = to_np(pose["left_ankle"])
            ra = to_np(pose["right_ankle"])
        else:
            leg_length = (rh[0] - lh[0]) * 2.2
            la = np.array([lh[0], lh[1] + leg_length])
            ra = np.array([rh[0], rh[1] + leg_length])
        lh, rh = expand_segment(lh, rh, 0.14)
        la, ra = expand_segment(la, ra, 0.10)
        return np.array([lh, rh, ra, la], dtype=np.float32)
