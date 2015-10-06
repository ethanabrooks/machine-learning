from numpy import loadtxt
from numpy.ma import sqrt
import matplotlib.pyplot as plt
import time

__author__ = 'Ethan'
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np

subdivision = 5
shift = 1
degree = 3
intervals_ = []
for p in range(-degree, degree):
    for m in range(subdivision):
        intervals_.append((1 + m) * 10 ** (p + shift) / float(subdivision))

def not_whole(x):
    return x - int(x) != 0

def intervals(start, stop, step):
    scalar = 1
    while not_whole(start) or not_whole(stop) or not_whole(step):
        start, stop, step = (s * 10 for s in (start, stop, step))
        scalar /= 10.
    start, stop, step = (int(s) for s in (start, stop, step))
    return [i * scalar for i in (range(start, stop, step))]

# param_grid = [
# {'C': intervals(.05, 10, .5), 'gamma': intervals(.5, 9, .5), 'kernel': ['rbf']}
# ]
# [0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06000000000000001,
# 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0,
# 40.0, 60.0, 80.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0]
# C:  8.0
# Gamma:  0.08
# Sigma:  2.5

    # {'C': intervals(1, 20, 1), 'gamma': intervals(.1, 2, .1), 'kernel': ['rbf']}
    # C:  6  Gamma:  0.1  Sigma:  2.2360679775

    # {'C': intervals(1, 10, .5), 'gamma': intervals(.01, .2, .01), 'kernel': ['rbf']}
# C:  3.6  Gamma:  0.14  Sigma:  1.88982236505

    # {'C': intervals(2, 5, .1), 'gamma': intervals(.1, .18, .002), 'kernel': ['rbf']}
# C:  3.6  Gamma:  0.135  Sigma:  1.9245008973

    # {'C': intervals(1, 4, .4), 'gamma': intervals(.08, .15, .001), 'kernel': ['rbf']}
    # {'C': intervals(2.6, 5.6, .2), 'gamma': intervals(.13, .14, .0005), 'kernel': ['rbf']}
    # C:  3.6  Gamma:  0.135  Sigma:  1.9245008973

    # {'C': intervals(.0001, 10, .035), 'gamma': intervals(.0001, 10, .035), 'kernel': ['rbf']}
    # C:  3.5351  Gamma:  0.1401  Sigma:  1.88914778985

param_grid = [
    {'C': intervals(3, 4, .0035), 'gamma': intervals(.0001, 1, .0035), 'kernel': ['rbf']}
]
# Load Data
filename = 'data/svmTuningData.dat'
data = loadtxt(filename, delimiter=',')
X = data[:, :2]
y = np.array(data[:, 2])
n, d = X.shape

start_time = time.clock()
clf = GridSearchCV(SVC(), param_grid)
clf.fit(X, y)
print "time:", time.clock() - start_time

C = clf.best_params_['C']
gamma = clf.best_params_['gamma']
sigma = sqrt(1 / (2 * gamma))
score = clf.best_score_
print "C: ", C, " Gamma: ", gamma, " Sigma: ", sigma, " Score: ", score


print "Testing the SVMs..."

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)

plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, predictions, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)  # Plot the training points
plt.title('Blah')
plt.axis('tight')

plt.show()
