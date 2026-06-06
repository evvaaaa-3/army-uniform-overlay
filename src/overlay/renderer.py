from src.overlay.warper import warp_to_body_quad
from src.overlay.compositor import alpha_composite_bgra_on_bgr


class UniformRenderer:
    def __init__(self, garment_library, body_mapper):
        self.garment_library = garment_library
        self.body_mapper = body_mapper

    def render(self, frame_bgr, pose, view_name):
        output = frame_bgr.copy()

        # Side view needs a different mapper.
        # Do not use front/back quad logic for side assets.
        if view_name in ["left", "right"]:
            return output

        shirt_quad = self.body_mapper.torso_quad(pose)
        trousers_quad = self.body_mapper.legs_quad(pose)

        # Render trousers first
        if trousers_quad is not None:
            trousers_asset = self.garment_library.get("trousers", view_name)
            trousers_overlay = warp_to_body_quad(
                trousers_asset,
                trousers_quad,
                output.shape,
            )
            output = alpha_composite_bgra_on_bgr(output, trousers_overlay, 0.92)

        # Render shirt second, so it sits over waistband
        if shirt_quad is not None:
            shirt_asset = self.garment_library.get("shirt", view_name)
            shirt_overlay = warp_to_body_quad(
                shirt_asset,
                shirt_quad,
                output.shape,
            )
            output = alpha_composite_bgra_on_bgr(output, shirt_overlay, 0.92)

        return output