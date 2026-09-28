"""The frozen train/validation split shared by every model.

It reproduces the 2021 notebook: images sorted by id, ``KFold(n_folds, shuffle=True,
random_state=seed)``, keep fold ``fold`` as validation. With the defaults (8 folds, fold 0, seed
42) the 2021 model is evaluated on the images it was validated on, not ones it was trained on.
"""

from sklearn.model_selection import KFold


def split_image_ids(
    image_ids: list[str], n_folds: int, fold: int, seed: int
) -> dict[str, list[str]]:
    """Split images into train and validation for one K-fold fold.

    Parameters
    ----------
    image_ids : list[str]
        Image ids, already sorted (the split depends on their order).
    n_folds : int
        Number of folds.
    fold : int
        Index of the fold kept for validation.
    seed : int
        Shuffle seed.

    Returns
    -------
    dict[str, list[str]]
        Image ids under ``"train"`` and ``"val"``.

    Raises
    ------
    ValueError
        If ``fold`` is not in ``range(n_folds)``.

    Examples
    --------
    >>> split = split_image_ids(["a", "b", "c", "d"], n_folds=2, fold=0, seed=0)
    >>> sorted(split["train"] + split["val"])
    ['a', 'b', 'c', 'd']
    """
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(image_ids)
    for index, (train_idx, val_idx) in enumerate(folds):
        if index == fold:
            return {
                "train": [image_ids[i] for i in train_idx],
                "val": [image_ids[i] for i in val_idx],
            }
    raise ValueError(f"fold {fold} out of range for {n_folds} folds")
