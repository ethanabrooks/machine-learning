from unittest import TestCase
from polyreg import learningCurve
import pytest
from sklearn.datasets import make_classification


__author__ = 'Ethan'


@pytest.fixture
def data():
    return make_classification(n_samples=3, n_features=1, n_informative=1,
                               n_redundant=0, n_classes=2,
                               n_clusters_per_class=1, random_state=0)


def test_learningCurve():
    pass

def get_error(model, X, y):
    pass