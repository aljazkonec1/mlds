import numpy as np
import scipy.optimize
from sklearn.metrics import log_loss
import random

class MultinomialLogReg:
    def __init__(self):
        self.coefficients = None

    def _softmax(self, X):
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True))
        return probs

    def _likelihood(self, theta, *args):
        X, y = args
        theta = theta.reshape(X.shape[1], -1)
        probabilities = self._softmax(np.dot(X, theta))
        log_likelihood = log_loss(y, probabilities)
        return np.float64(log_likelihood) + 0.2 * np.sum(theta ** 2)

    def build(self, X, y):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        num_features = X.shape[1]
        num_classes = len(np.unique(y))
        initial_theta = np.random.rand(num_features * (num_classes))

        result = scipy.optimize.minimize(self._likelihood, initial_theta, args=(X, y), method='L-BFGS-B', options={'disp': False})
        self.coefficients = (result.x).reshape(num_features, num_classes)
        return self

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        probabilities = self._softmax(np.dot(X, self.coefficients))
        return probabilities