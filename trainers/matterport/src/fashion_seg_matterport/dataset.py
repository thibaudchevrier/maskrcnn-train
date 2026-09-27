"""Prepared iMaterialist annotations exposed as a Matterport ``utils.Dataset``."""

from pathlib import Path

import numpy as np
import polars as pl
from fashion_seg_core import rle
from mrcnn import utils

SOURCE = "fashion"


class FashionDataset(utils.Dataset):
    """Images are read lazily from ``image_dir``; masks are decoded from their RLE on demand."""

    def __init__(self, records: pl.DataFrame, image_dir: str | Path, class_names: list[str]):
        super().__init__()
        # Model class ids: 0 is the background, dataset category k is class k + 1.
        for class_id, name in enumerate(class_names[1:], start=1):
            self.add_class(SOURCE, class_id, name)
        for row in records.iter_rows(named=True):
            self.add_image(
                SOURCE,
                image_id=row["image_id"],
                path=str(Path(image_dir) / f"{row['image_id']}.jpg"),
                height=int(row["height"]),
                width=int(row["width"]),
                categories=[int(c) for c in row["class_ids"]],
                rles=list(row["rles"]),
            )
        self.prepare()

    def load_mask(self, image_id: int) -> tuple[np.ndarray, np.ndarray]:
        """[H, W, N] bool masks and their [N] model class ids."""
        info = self.image_info[image_id]
        height, width = info["height"], info["width"]
        masks = np.stack([rle.decode(r, height, width) for r in info["rles"]], axis=-1)
        return masks, np.array([c + 1 for c in info["categories"]], dtype=np.int32)

    def image_reference(self, image_id: int) -> str:
        return self.image_info[image_id]["path"]
