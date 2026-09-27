import numpy as np

from fashion_seg import rle


def test_decode_is_one_indexed_column_major():
    # Pixels 1-2 are the first two rows of column 0; pixel 4 is row 0 of column 1.
    mask = rle.decode("1 2 4 1", height=3, width=2)
    expected = np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_roundtrip_random_masks():
    rng = np.random.default_rng(0)
    for _ in range(20):
        mask = rng.random((17, 11)) > 0.6
        np.testing.assert_array_equal(rle.decode(rle.encode(mask), 17, 11), mask)


def test_empty_and_full_masks():
    assert rle.encode(np.zeros((4, 4), dtype=bool)) == ""
    assert not rle.decode("", 4, 4).any()
    assert rle.encode(np.ones((4, 4), dtype=bool)) == "1 16"
