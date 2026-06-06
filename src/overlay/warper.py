import cv2
import numpy as np


def _anchor_point(asset, name):
    x_norm, y_norm = asset.anchors[name]
    return [x_norm * asset.width, y_norm * asset.height]


def source_quad_from_anchors(asset):
    anchors = asset.anchors

    if all(k in anchors for k in ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]):
        return np.array(
            [
                _anchor_point(asset, "left_shoulder"),
                _anchor_point(asset, "right_shoulder"),
                _anchor_point(asset, "right_hip"),
                _anchor_point(asset, "left_hip"),
            ],
            dtype=np.float32,
        )

    if all(k in anchors for k in ["left_hip", "right_hip", "right_ankle", "left_ankle"]):
        return np.array(
            [
                _anchor_point(asset, "left_hip"),
                _anchor_point(asset, "right_hip"),
                _anchor_point(asset, "right_ankle"),
                _anchor_point(asset, "left_ankle"),
            ],
            dtype=np.float32,
        )

    raise ValueError(
        f"Unsupported anchors for {asset.garment_type}/{asset.view_name}: {anchors.keys()}"
    )


def warp_to_body_quad(asset, dst_quad, frame_shape):
    H, W = frame_shape[:2]

    src_quad = source_quad_from_anchors(asset)
    matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

    warped = cv2.warpPerspective(
        asset.image_bgra,
        matrix,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    return warped