from mapFeature import mapFeature, mapFeatureK
import numpy as np
import numpy.testing as npt

__author__ = 'Ethan'


def str_to_matrix(string):
    rows = string[2:-2].split('] [')
    arrays = [np.fromstring(row, sep=' ') for row in rows]
    return np.vstack(arrays)


def test_string_to_matrix():
    npt.assert_equal(str_to_matrix("[[1   2] [3   4]]"), np.matrix([[1, 2], [3, 4]]))


def test_mapFeatureK():
    x1 = np.matrix([2, -1]).T
    x2 = np.matrix([3, 2]).T
    actual = mapFeatureK(x1, x2, 1)
    desired = np.matrix([[1, 3, 2, 6], [1, 2, -1, -2]])
    npt.assert_almost_equal(actual, desired)

    x2 = np.matrix(3)
    x1 = np.matrix(2)
    actual = mapFeatureK(x1, x2, 1)
    desired = np.matrix([1, 3, 2, 6])
    npt.assert_almost_equal(actual, desired)

    x2 = np.matrix(3)
    x1 = np.matrix(2)
    actual = mapFeatureK(x1, x2, 2)
    desired = np.matrix([1, 3, 9, 2, 6, 18, 4, 12, 36])
    npt.assert_almost_equal(actual, desired)

def test_mapFeature():
    x1 = np.matrix(2)
    x2 = np.matrix(3)
    assert mapFeature(x1, x2) is not None
