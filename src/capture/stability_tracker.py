import math


class StabilityTracker:
    def __init__(self, required_stable_seconds=2.0, max_motion_px=18):
        self.required_stable_seconds = required_stable_seconds
        self.max_motion_px = max_motion_px
        self.landmarks = [
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.previous_points = None
        self.stable_since = None

    def reset(self):
        self.previous_points = None
        self.stable_since = None

    def update(self, pose, timestamp):
        if pose is None:
            self.reset()
            return {"stable": False, "stable_for": 0.0, "motion": 0.0}

        points = {}
        for name in self.landmarks:
            point = pose.get(name)
            if point is None or point.get("vis", 0.0) <= 0:
                self.reset()
                return {"stable": False, "stable_for": 0.0, "motion": 0.0}
            points[name] = (float(point["x"]), float(point["y"]))

        motion = 0.0
        if self.previous_points is not None:
            distances = []
            for name in self.landmarks:
                x1, y1 = self.previous_points[name]
                x2, y2 = points[name]
                distances.append(math.hypot(x2 - x1, y2 - y1))
            motion = sum(distances) / len(distances)

        self.previous_points = points

        if motion > self.max_motion_px:
            self.stable_since = timestamp
            return {"stable": False, "stable_for": 0.0, "motion": motion}

        if self.stable_since is None:
            self.stable_since = timestamp

        stable_for = max(0.0, timestamp - self.stable_since)
        return {
            "stable": stable_for >= self.required_stable_seconds,
            "stable_for": stable_for,
            "motion": motion,
        }
