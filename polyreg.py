'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np
import numpy.testing as npt


# -----------------------------------------------------------------
# Class PolynomialRegression
# -----------------------------------------------------------------


class PolynomialRegression:
    def __init__(self, degree=1, regLambda=1E-8):
        """
        Constructor
        """
        # TODO
        self.regLambda = regLambda
        self.degree = degree

    def polyfeatures(self, X, degree):  # QUESTION can we get
        # rid of degree and use the attribute?
        """
        Expands the given X into an n * d array of polynomial features of
            degree d.

        :param degree:
        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        """
        # TODO
        polyX = np.matrix(X)
        polyX.shape = (1, len(X))
        for i in range(1, degree):
            Xi = np.matrix(np.multiply(polyX[-1, :], X))
            polyX = np.vstack((polyX, Xi))
        return polyX.T

    def set_stats(self, matrix):
        self.means, self.stdevs = (array for array in (matrix.mean(0), matrix.std(0)))

    def rescale(self, matrix):
        rescaled = matrix - self.means.repeat(len(matrix), axis=0)
        std_list = self.stdevs.tolist()[0]
        for i, stdev in enumerate(std_list):
            if self.stdevs[:, i] != 0:
                rescaled[:, i] = rescaled[:, i] / stdev
        ones_column = np.ones(((len(matrix)), 1))
        return np.hstack((ones_column, rescaled))

    def get_theta(self, Xex, y):
        assert y.size == len(Xex)
        assert Xex.shape[1] == self.degree + 1
        y = np.matrix(y).reshape(1, len(Xex))

        regMatrix = self.regLambda * np.eye(self.degree + 1)
        regMatrix[0, 0] = 0
        return y * Xex * np.linalg.pinv(Xex.T * Xex + regMatrix)

    def fit(self, X, y):
        """
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        """
        # TODO
        polyX = self.polyfeatures(X, self.degree)
        self.set_stats(polyX)
        rescaled = self.rescale(polyX)
        self.theta = self.get_theta(rescaled, y)

    def predict(self, X):
        """
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        """
        # TODO
        polyX = self.polyfeatures(X, self.degree)
        return np.squeeze(np.asarray(self.theta * self.rescale(polyX).T))


# -----------------------------------------------------------------
# End of Class PolynomialRegression
# -----------------------------------------------------------------

def get_error(model, X, y):
    predictions = model.predict(X)
   # TODO
    sq = np.vectorize(lambda x: x**2)
    errors = sq(predictions - y)
    npt.assert_almost_equal(sum(errors)/len(y), errors.mean())
    return errors.mean()


def get_error_sets(train, test, n, model):
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))

    for i in range(n):
        train_i = [data[:i + 1] for data in train]
        model.fit(*train_i)
        errorTrain[i] = get_error(model, *train)
        errorTest[i] = get_error(model, *test)

    return errorTrain, errorTest


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    """
    Compute learning curve

    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree

    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]

    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start
        displaying the learning curve at n = 2 (or higher)
    """

    # TODO -- complete rest of method;
    # errorTrain and errorTest are already the correct shape

    n = len(Xtrain)
    model = PolynomialRegression(degree, regLambda)
    return get_error_sets((Xtrain, Ytrain), (Xtest, Ytest), n, model)



