import numpy as np

from src.features import combine_pair


def test_combine_pair_symmetric():
    a = np.array([0, 1, 0, 1, 1, 0, 0, 0] * 4, dtype=np.uint8)
    b = np.array([1, 0, 0, 1, 0, 1, 1, 0] * 4, dtype=np.uint8)
    feat_ab = combine_pair(a, b)
    feat_ba = combine_pair(b, a)
    assert np.array_equal(feat_ab, feat_ba)


def test_combine_pair_shape():
    a = np.zeros(2048, dtype=np.uint8)
    b = np.ones(2048, dtype=np.uint8)
    out = combine_pair(a, b)
    assert out.shape == (3 * 2048,)
    assert out.dtype == np.float32
