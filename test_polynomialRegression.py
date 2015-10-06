from unittest import TestCase
import numpy as np
import pytest
from polyreg import PolynomialRegression, get_error, learningCurve
import numpy.testing as npt


__author__ = 'Ethan'


def str_to_matrix(string):
    rows = string[2:-2].split('] [')
    arrays = [np.fromstring(row, sep=' ') for row in rows]
    return np.vstack(arrays)


def test_string_to_matrix():
    npt.assert_equal(str_to_matrix("[[1   2] [3   4]]"), np.matrix([[1, 2], [3, 4]]))


@pytest.fixture
def allData():
    # load the data
    filePath = "data/polydata.dat"
    file = open(filePath, 'r')
    return np.loadtxt(file, delimiter=',')


@pytest.fixture
def X(allData):
    return allData[:6, 0]


@pytest.fixture
def y(allData):
    return allData[:6, 1]


@pytest.fixture
def polyreg():
    return PolynomialRegression(3)


@pytest.fixture
def polyX(polyreg, X):
    return polyreg.polyfeatures(X, polyreg.degree)


def test_polyfeatures(polyreg, X):
    npt.assert_equal(polyreg.polyfeatures(X, polyreg.degree), str_to_matrix(
        "[[   0.       0.       0.   ] "
        "[   0.5      0.25     0.125] "
        "[   1.       1.       1.   ] "
        "[   2.       4.       8.   ] "
        "[   3.       9.      27.   ] "
        "[   5.      25.     125.   ]]"
    ))


def test_rescale(polyreg, polyX):
    n, d = polyX.shape
    polyreg.set_stats(polyX)
    rescaled = polyreg.rescale(polyX)
    for i in range(n):
        for j in range(d):
            mean = polyX[:, j].mean()
            stdev = polyX[:, j].std()
            desired = (polyX[i, j] - mean) / stdev
            npt.assert_almost_equal(rescaled[i, j + 1], desired)
    desired = str_to_matrix("[[1  -1.13175601 -0.7420452  -0.59818533] " \
                            "[1  -0.83651531 -0.71368678 -0.59540091] " \
                            "[1  -0.54127461 -0.62861154 -0.57591   ] " \
                            "[1  0.04920678 -0.28831055 -0.41998272] " \
                            "[1  0.63968818  0.27885775  0.00324849] " \
                            "[1  1.82065098  2.09379632  2.18623048]]")
    npt.assert_almost_equal(rescaled, desired)


def test_get_theta(polyreg, polyX, y):
    polyreg.set_stats(polyX)
    X = polyreg.rescale(polyX)
    theta = polyreg.get_theta(X, y)
    npt.assert_almost_equal(theta * X.T * X, y * X)


def test_predict(polyreg, X, y):
    polyreg.fit(X, y)
    actual = polyreg.predict(X)
    desired = np.squeeze(str_to_matrix(
        "[[ 1.79599233  2.54434227  3.17149473  4.06310245  4.47260597  3.85246225]]"
    ))
    npt.assert_almost_equal(actual, desired)


@pytest.fixture
def polydata():
    filePath = "data/polydata.dat"
    file = open(filePath, 'r')
    return np.loadtxt(file, delimiter=',')


@pytest.fixture
def Xp(polydata):
    return polydata[:, 0]


@pytest.fixture
def yp(polydata):
    return polydata[:, 1]


@pytest.fixture
def X_train(Xp):
    return Xp[np.arange(1, 10)]


@pytest.fixture
def y_train(yp):
    return yp[np.arange(1, 10)]


@pytest.fixture
def X_test(Xp):
    return Xp[np.zeros(1, dtype=int)]


@pytest.fixture
def y_test(yp):
    return yp[np.zeros(1, dtype=int)]


def test_get_error(polyreg, X, y, X_train, y_train, X_test, y_test):
    polyreg.fit(X_train, y_train)
    actual = get_error(polyreg, X_test, y_test)
    npt.assert_almost_equal(actual, 0.13167431777)
    polyreg.fit(X, y)
    actual = get_error(polyreg, X, y)
    npt.assert_almost_equal(actual, 0.198073827773)


def test_learningCurve(X_train, y_train, X_test, y_test):
    regLambda = 1
    degree = 4
    desired = (
        np.array([0.72777778, 0.72777778, 49.51268674, 5.30342444,
                  0.93613619, 0.40144971, 0.29421168, 0.24684666, 0.24916648]),
        np.array([0.25, 0.25, 0.09617399, 0.18817116, 0.39917127,
                  0.47896523, 0.50556957, 0.53004403, 0.55796313])
    )
    actual = learningCurve(X_train, y_train, X_test, y_test, regLambda, degree)
    npt.assert_almost_equal(actual, desired)
