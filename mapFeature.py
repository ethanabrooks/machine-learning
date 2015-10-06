from itertools import product
import numpy as np


def mapFeature(x1, x2):
    """
    Maps the two input features to quadratic features.

    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial

    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    """
    return mapFeatureK(x1, x2, 6)


def mapFeatureK(x1, x2, k):
    degrees = product(range(k + 1), repeat=2)
    raise_to_the = np.vectorize(lambda k, x: x ** k, excluded=['k'])
    column_list = [
        np.multiply(raise_to_the(i, x1), raise_to_the(j, x2))
        for i, j in degrees
    ]
    return np.column_stack(column_list)



