import numpy as np


def alpha_composite_bgra_on_bgr(frame_bgr, overlay_bgra, alpha_multiplier=1.0):
    if overlay_bgra is None:
        return frame_bgr

    alpha = overlay_bgra[:, :, 3:4].astype(np.float32) / 255.0
    alpha *= alpha_multiplier

    foreground = overlay_bgra[:, :, :3].astype(np.float32)
    background = frame_bgr.astype(np.float32)

    output = foreground * alpha + background * (1.0 - alpha)
    return output.astype(np.uint8)