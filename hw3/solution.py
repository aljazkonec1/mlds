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
        log_likelihood = -np.sum(np.log(probabilities[np.arange(X.shape[0]), y]))
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
    

class OrdinalLogReg:
    def __init__(self):
        self.coefficients = None
        self.thresholds = None

    def _cumulative_proba(self, X):
        return 1 / (1 + np.exp(X))

    def _likelihood(self, theta, X, y):
        
        nr_thresholds = len(np.unique(y)) - 2
        thresholds = np.hstack((-np.inf, 0, np.cumsum(theta[0:nr_thresholds]), np.inf))

        probabilities = self._cumulative_proba(thresholds[:, np.newaxis] - np.dot(X, theta[nr_thresholds:]))

        class_probabilities = (probabilities[:-1, :] - probabilities[1:, :]).T
        mask = np.zeros_like(class_probabilities, dtype=bool)
        mask[np.arange(X.shape[0]), y] = True

        log_likelihood = -np.sum(np.log(class_probabilities[mask])) - np.sum(np.log(1 - class_probabilities[mask == 0]))
        return log_likelihood + 0.1 * np.sum(theta ** 2)

    def build(self, X, y):
        # X = np.hstack((np.ones((X.shape[0], 1)), X))
        num_features = X.shape[1]
        initial_theta = np.hstack((np.ones(len(np.unique(y)) - 2), np.random.rand(num_features)))
        
        result = scipy.optimize.minimize(self._likelihood, initial_theta, args=(X, y), method='L-BFGS-B', options={'disp': False})
        r = result.x
        nr_th = len(np.unique(y)) - 2
        self.thresholds = np.hstack((-np.inf, 0, np.cumsum(r[0:nr_th]), np.inf))
        self.coefficients = r[nr_th:]
        return self

    def predict(self, X):

        b = self.thresholds[:, np.newaxis] - np.dot(X, self.coefficients)
        probabilities = self._cumulative_proba(b)
        probabilities = (probabilities[:-1, :] - probabilities[1:, :]).T

        return probabilities
    


MBOG_TRAIN = 50

def multinomial_bad_ordinal_good(n, rand=None):
    if rand is None:
        np.random.seed()
    else:
        np.random.seed(rand.randint(0, 2**31))

    X = np.random.normal(loc= 0, scale=1, size=(n, 12))
    y = np.random.randint(0, 3, size=n)

    return X, y

def test_mult_bad_ord_good():

    X, y = multinomial_bad_ordinal_good(MBOG_TRAIN)
    X_test, y_test = multinomial_bad_ordinal_good(1000)

    l = MultinomialLogReg()
    l.build(X, y)

    pred = l.predict(X_test)
    # pred = np.argmax(pred, axis=1)

    o = OrdinalLogReg()
    o.build(X, y)
    pred_o = o.predict(X_test)

    l_loss = log_loss(y_test, l.predict(X_test))
    o_loss = log_loss(y_test, o.predict(X_test))



    print(l_loss, o_loss)
    return l_loss, o_loss


if __name__ == "__main__":
    a = 0
    # X = np.array([[0, 0],
    #                [0, 1],
    #                [1, 0],
    #                [1, 1],
    #                [1, 1]])
    # y = np.array([0, 0, 1, 1, 2])
    # train = X[::2], y[::2]
    # test = X[1::2], y[1::2]
    # l = OrdinalLogReg()

    # theta = np.random.rand(2, 3)

    # l.build(X, y)
    # print(l.coefficients)
    # print(l.thresholds)
    # print(test[0])
    # prob = l.predict(test[0])
    # print(prob)
    # print(prob.shape)
    # print(prob.sum(axis=1))
    # print((prob <= 1).all())
    # print((prob >= 0).all())

