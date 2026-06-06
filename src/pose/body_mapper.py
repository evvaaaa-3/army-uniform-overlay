import numpy as np

from src.utils.geometry import to_np, midpoint, expand_segment


class BodyMapper:
    """
    Converts detected pose landmarks into dynamic body regions.

    Person-agnostic:
    no shoulder, hip, knee, ankle position is hardcoded for a specific person.
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
        """
        Shirt destination quad:
        left shoulder, right shoulder, right hip, left hip.

        Smaller expansion than before because the asset already includes sleeves
        and garment looseness.
        """
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

        # Smaller universal expansion.
        # These are garment-fit constants, not person-specific values.
        ls, rs = expand_segment(ls, rs, 0.06)
        lh, rh = expand_segment(lh, rh, 0.04)

        # Slight shirt extension below hip.
        lh = lh + torso_vec * 0.08
        rh = rh + torso_vec * 0.08

        return np.array([ls, rs, rh, lh], dtype=np.float32)

    def legs_quad(self, pose):
        """
        Trouser destination quad:
        left hip, right hip, right ankle, left ankle.
        """
        needed = ["left_hip", "right_hip", "left_ankle", "right_ankle"]

        if not self.has_points(pose, needed):
            return None

        lh = to_np(pose["left_hip"])
        rh = to_np(pose["right_hip"])
        la = to_np(pose["left_ankle"])
        ra = to_np(pose["right_ankle"])

        # Smaller expansion to avoid oversized trousers.
        lh, rh = expand_segment(lh, rh, 0.08)
        la, ra = expand_segment(la, ra, 0.08)

        return np.array([lh, rh, ra, la], dtype=np.float32)