"""Prepared iMaterialist annotations exposed as a Matterport ``utils.Dataset``."""

from pathlib import Path

import numpy as np
import polars as pl
from fashion_seg_contract import rle
from mrcnn import utils

SOURCE = "fashion"


class FashionDataset(utils.Dataset):
    """Prepared iMaterialist annotations exposed as a Matterport ``utils.Dataset``.

    Images are read lazily from ``image_dir``; masks are decoded from their RLE on demand.

    Parameters
    ----------
    records : pl.DataFrame
        Prepared annotations (see ``fashion_seg.data.annotations``) of the images to include.
    image_dir : str | Path
        Directory of the ``<image_id>.jpg`` files.
    class_names : list[str]
        Class names indexed by model class id (0 = background).
    """

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
        """Decode the instance masks of an image.

        Parameters
        ----------
        image_id : int
            Internal image index of the dataset (not the iMaterialist image id).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``[height, width, N]`` bool masks and their ``[N]`` model class ids (category + 1).
        """
        info = self.image_info[image_id]
        height, width = info["height"], info["width"]
        masks = np.stack([rle.decode(r, height, width) for r in info["rles"]], axis=-1)
        return masks, np.array([c + 1 for c in info["categories"]], dtype=np.int32)
