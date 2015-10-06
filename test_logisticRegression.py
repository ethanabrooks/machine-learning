from unittest import TestCase
from logreg import LogisticRegression
import pytest
import numpy as np
import numpy.testing as npt

__author__ = 'Ethan'


@pytest.fixture
def allData():
    # load the data
    filePath = "data/data1.dat"
    file = open(filePath, 'r')
    return np.loadtxt(file, delimiter=',')


@pytest.fixture
def X(allData):
    return allData[:15, :-1]


@pytest.fixture
def y(allData):
    return allData[:15, -1]


@pytest.fixture
def logreg():
    return LogisticRegression()

def test_gradientJ(logreg):
    X = np.matrix(([1], [1]))
    y = X
    logreg.theta = np.matrix([3.01131387, 1.20161574])
    actual = logreg.gradientJ(X, y, 0, 2, .01, logreg.theta)
    desired = np.matrix([0.00093912])
    npt.assert_almost_equal(actual, desired)

def test_gradient0(logreg):
    logreg.theta = np.matrix([5, -2])
    X = np.matrix([8])
    y = np.array([0])
    actual = logreg.gradient0(X, y, logreg.theta)
    desired = np.matrix([0.0000167014])
    npt.assert_almost_equal(actual, desired)

    logreg.theta = np.matrix([1, 1])
    X = np.matrix([1])
    y = np.array([1])
    actual = logreg.gradient0(X, y, logreg.theta)
    desired = np.matrix([-0.11920292202211755])
    npt.assert_almost_equal(actual, desired)


def test_Gsummand(logreg):
    logreg.theta = np.matrix([5, -2])
    actual = logreg.Gsummand([8], 1, logreg.theta, 0)
    desired = -7.99986638863
    npt.assert_almost_equal(actual, desired)

    logreg.theta = np.matrix([15, -4])
    actual = logreg.Gsummand([5], 0, logreg.theta, 0)
    desired = 0.0334642546214
    npt.assert_almost_equal(actual, desired)

def test_computeCost(logreg):
    logreg.theta = np.matrix([1, 1])
    X = np.matrix([1])
    y = np.array([0])
    actual = logreg.computeCost(logreg.theta, X, y, logreg.regLambda)
    desired = 2.1339990788548366
    npt.assert_almost_equal(actual, desired)

    logreg.theta = np.matrix([5, -2])
    X = np.matrix([8])
    y = np.array([0])
    actual = logreg.computeCost(logreg.theta, X, y, logreg.regLambda)
    desired = 0.026942525597
    npt.assert_almost_equal(actual, desired)

    logreg.theta = np.matrix([15, -4])
    X = np.matrix([7])
    y = np.array([1])
    actual = logreg.computeCost(logreg.theta, X, y, logreg.regLambda)
    desired = 13.0776231338
    npt.assert_almost_equal(actual, desired)


def test_predict(logreg, X):
    n, d = X.shape
    logreg.theta = np.matrix(np.zeros((1, d+1)))
    assert logreg.predict(X[0, :])[0, 0] == .5

    logreg.theta = np.matrix([0, 1, 2])
    actual = logreg.predict([4, -3])
    desired = np.matrix([0.11920292])
    npt.assert_almost_equal(actual, desired)


def test_Jsummand(logreg):
    logreg.theta = np.matrix([5, -2])
    actual = logreg.Jsummand(1, [2, 8], logreg.theta)
    desired = 9223372036854775807
    npt.assert_almost_equal(actual, desired)

    logreg.theta = np.matrix([15, -4])
    actual = logreg.Jsummand(0, [2, 5], logreg.theta)
    desired = 9223372036854775807
    npt.assert_almost_equal(actual, desired)

