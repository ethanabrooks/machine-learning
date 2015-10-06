"""
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
"""
from copy import copy
from math import exp, log
import sys
import numpy as np
import numpy.linalg as npl


class LogisticRegression:
    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        """
        Constructor
        """
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters

    def Jsummand(self, yi, xi, theta):
        try:
            if yi:
                return -log(h(theta, xi))
            else:
                return -log(1 - h(theta, xi))
        except ValueError:
            return sys.maxint


    def computeCost(self, theta, X, y, regLambda):
        """
        Computes the objective function
        :param theta:
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        """

        n, _ = X.shape
        summation = [self.Jsummand(y[i], X[i, :], theta) for i in range(n)]
        return sum(summation) + regLambda * npl.norm(theta) / 2.

    def gradient0(self, X, y, theta):
        n = X.shape[0]
        return sum((h(theta, X[i, :]) - y[i]) for i in range(n))

    def Gsummand(self, Xi, yi, theta, j):
        return (
                   h(theta, Xi) - yi
               )[0, 0] * Xi[0, j - 1]  # because of added ones column

    def gradientJ(self, X, y, j, n, regLambda, theta):
        return sum(
            self.Gsummand(X[i, :], y[i], theta, j) for i in range(n)
        ) + regLambda * theta[0, j]

    def computeGradient(self, theta, X, y, regLambda):
        """
        Computes the gradient of the objective function
        :param regLambda:
        :param theta:
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        """
        n, d = X.shape
        assert n == len(y)
        # assert theta.shape == (1, d)

        gradient = [self.gradient0(X, y, theta)[0, 0]]
        for j in range(1, d):
            gradient.append(self.gradientJ(X, y, j, n, regLambda, theta))
        return np.matrix(gradient)


    def fit(self, X, y):
        """
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        """

        X = np.matrix(X)
        ones_column = np.ones((X.shape[0], 1))
        X = np.column_stack([ones_column, X])
        n, d = X.shape

        # self.theta = np.matrix([4.88002867, 1.04755767, -0.06604604,
        # -0.08645819, -0.00889369, 0.16846109,
        #                         0.05923942, -0.04213808, 0.0160093, 0.54581483])
        # self.theta = np.matrix([1.71297509, 3.98109885, 3.7138258])
        # self.theta = np.matrix([3.01131387, 1.20161574])
        # self.theta = np.matrix([3.81498153, 2.48674634, 0.25994663, -3.94751691,
        # -0.36834352, 1.03430205, -1.96120589, -1.57746805,
        #                         -1.26575392, -3.2139977, 0.28956062, 0.27853624,
        #                         -0.21059673, -0.56927184, 0.27781856, 1.4877498, 0.97117391])
        self.theta = np.matrix(np.random.rand(1, d))
        i = 0
        while True:
            i += 1
            cost = self.computeCost(self.theta, X, y, self.regLambda)
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            print \
                "Iteration: ", i, \
                " Cost: ", cost, \
                " Theta: ", self.theta
            # " Gradient: ", gradient
            # " Delta: ", delta, \
            theta_old = self.theta.copy()
            self.theta += -self.alpha * gradient
            delta = npl.norm(self.theta - theta_old)
            if delta <= self.epsilon:
                break


    def predict(self, X):
        """
            Used the model to predict values for each instance in X
            Arguments:
                X is a n-by-d numpy matrix
            Returns:
                an n-dimensional numpy vector of the predictions
            """
        return h(self.theta, X)

        # X = np.matrix(X)
        # ones_column = np.ones((X.shape[0], 1))
        # X = np.column_stack([ones_column, X])
        #
        # def sigma_single_val(z):
        # try:
        #         return 1. / (1. + exp(-z))
        #     except OverflowError:
        #         if z < 0:
        #             return 0.
        #         else:
        #             return 1.
        #
        # sigma = np.vectorize(sigma_single_val)
        # z = X * self.theta.T
        # np_asarray = np.asarray(sigma(z))
        # return np_asarray


def h(theta, X):
    theta = np.copy(theta)
    theta.shape = (theta.size, 1)

    def sigma_single_val(z):
        try:
            return 1. / (1. + exp(-z))
        except OverflowError:
            if z < 0:
                return 0.
            else:
                return 1.

    sigma = np.vectorize(sigma_single_val)
    z = X * theta
    np_asarray = np.asarray(sigma(z))
    return np_asarray

