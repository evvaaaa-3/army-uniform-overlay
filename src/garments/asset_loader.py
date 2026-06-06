from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np


@dataclass
class GarmentView:
    garment_type: str
    view_name: str
    image_bgra: np.ndarray
    width: int
    height: int
    anchors: dict


class GarmentLibrary:
    def __init__(self, asset_root: Path):
        self.asset_root = Path(asset_root)
        self.meta_path = self.asset_root / "garment_metadata.json"

        if not self.meta_path.exists():
            raise FileNotFoundError(f"Missing garment metadata: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.cache = {}
        self._load_all()

    def _load_bgra(self, path: Path):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            a = np.ones_like(b, dtype=np.uint8) * 255
            img = cv2.merge([b, g, r, a])

        return self._trim_alpha(img)

    def _trim_alpha(self, img_bgra):
        alpha = img_bgra[:, :, 3]
        ys, xs = np.where(alpha > 10)

        if len(xs) == 0 or len(ys) == 0:
            return img_bgra

        return img_bgra[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def _load_all(self):
        garments = self.metadata["garments"]

        for garment_type, garment_views in garments.items():
            self.cache[garment_type] = {}

            for view_name, view_data in garment_views.items():
                img_path = self.asset_root / view_data["file"]
                img_bgra = self._load_bgra(img_path)

                h, w = img_bgra.shape[:2]

                self.cache[garment_type][view_name] = GarmentView(
                    garment_type=garment_type,
                    view_name=view_name,
                    image_bgra=img_bgra,
                    width=w,
                    height=h,
                    anchors=view_data["anchors"],
                )

    def get(self, garment_type: str, view_name: str):
        return self.cache[garment_type][view_name]

    def available_garments(self):
        return list(self.cache.keys())

    def available_views(self, garment_type: str):
        return list(self.cache[garment_type].keys())