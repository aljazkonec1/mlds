import numpy as np
import csv
import math
from scipy.optimize import fmin_l_bfgs_b, minimize
from sklearn.metrics import mean_squared_error, log_loss
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit
import time
import pandas as pd
import sys


class ANN:
    def __init__(self, units, lambda_):
        self.units = units #number of neurons per hidden layer
        self.lambda_ = lambda_
        self.weight = []
        self.biases = []
        self.weight_shapes = []
        self.biases_shapes = []

        self.activations = []
        self.zs = []
        self.activation_function = self.sigmoid

    def initialize_weight(self, s= None):
        np.random.seed(s)
        self.weight = [np.random.randn( x, y) for x, y in zip(self.units[:-1], self.units[1:])]
        # self.weight = [np.random.randn( y, x) for x, y in zip(self.units[:-1], self.units[1:])]
        self.biases = [np.random.randn(1, y) for y in self.units[1:]]
        # self.biases = [np.random.randn(y, 1) for y in self.units[1:]]
        self.weight_shapes = [(y, x) for x, y in zip(self.units[:-1], self.units[1:])]
        self.biases_shapes = [(y, 1) for y in self.units[1:]]

    def sigmoid(self, x):
        try:
            return expit(x)
        except:
            print(x)
            print("zs ", self.zs)
            print("activations ", self.activations)
            print("weights ", self.weight)
            sys.exit(1)


        # return 1 / (1 + np.exp(-x))
        # return expit(x)
        
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
        # return expit(x) * (1 - expit(x))
    
    def cost_derivative(self, output_activations, y):
        return output_activations - y
    
    def hidden_layer_feedforward(self, x): # feed forward one element, do not compute output layer
        self.activations.append(x)

        for i in range(len(self.weight)-1):
            zi = np.dot(self.weight[i], self.activations[i]) + self.biases[i].flatten()
            self.zs.append(zi)
            self.activations.append(self.activation_function(zi))
    
    def feedforward(self, x):
        return NotImplementedError
    
    def loss(self, params, X, y):
        return NotImplementedError
    
    def fit(self, X, y):
        return NotImplementedError
    
    def predict(self, X):
        return NotImplementedError
    
    def log_loss2(self, y, preds): 
        return np.log(np.sum(preds * y))

    def backprop(self, x, y):
        self.activations = []   
        self.zs = []
        self.feedforward(x)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        delta = self.cost_derivative(self.activations[-1], y) 
        # * self.sigmoid_derivative(self.zs[-1]) #BP1


        nabla_b[-1] = delta #BP3
        
        nabla_w[-1] = np.dot(np.array([delta]).T, np.array([self.activations[-2]])) #BP4

        for l in range(2, len(self.units)):
            delta = np.dot(self.weight[-l+1].T, delta) * self.sigmoid_derivative(self.zs[-l]) #BP2
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(np.array([delta]).T, np.array([self.activations[-l-1]]))

        return nabla_b, nabla_w


    def gradient(self, params, X, y):
        self.set_params(params)

        nabla_b = [np.zeros(b).T for b in self.biases_shapes]
        nabla_w = [np.zeros(w) for w in self.weight_shapes]

        for x, y in zip(X, y): # for each element in the batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        nabla_b = [nb / X.shape[0] for nb in nabla_b]
        nabla_w = [nw / X.shape[0] for nw in nabla_w]
        nabla_w = [nw + 2 * self.lambda_ * w for nw, w in zip(nabla_w, self.weight)]
        return np.concatenate([nw.flatten() for nw in nabla_w] + [nb.flatten() for nb in nabla_b])

    def compute_numerical_gradient(self, x, y, epsilon=1e-6):
        numerical_nabla_b = [np.zeros(b).T for b in self.biases_shapes]
        numerical_nabla_w = [np.zeros(w) for w in self.weight_shapes]


        for i in range(len(self.weight)):
            for j in range(self.weight[i].shape[0]):
                for k in range(self.weight[i].shape[1]):
                    self.weight[i][j, k] += epsilon #za epsilon desno
                    cost_plus = self.loss(self.flatten_params(), x, y)[0]
                    self.weight[i][j, k] -= 2 * epsilon #za epsilon desno levo
                    cost_minus = self.loss(self.flatten_params(), x, y)[0]
                    self.weight[i][j, k] += epsilon
                    numerical_nabla_w[i][j, k] += (cost_plus - cost_minus) / (2 * epsilon)

        for i in range(len(self.biases)):
            for j in range(self.biases[i].shape[0]):
                self.biases[i][j, 0] += epsilon
                cost_plus = self.loss(self.flatten_params(), x, y)[0]
                self.biases[i][j, 0] -= 2 * epsilon
                cost_minus = self.loss(self.flatten_params(), x, y)[0]
                self.biases[i][j, 0] += epsilon

                numerical_nabla_b[i][0, j] += (cost_plus - cost_minus) / (2 * epsilon)

        numerical_nabla_b = [nb / x.shape[0] for nb in numerical_nabla_b]
        numerical_nabla_w = [nw / x.shape[0] for nw in numerical_nabla_w]

        numerical_nabla_w = [nw + 2 * self.lambda_ * w for nw, w in zip(numerical_nabla_w, self.weight)]
        return np.concatenate([nw.flatten() for nw in numerical_nabla_w] + [nb.flatten() for nb in numerical_nabla_b])    

    def optimize(self, X, y, epochs= 1):
        initial_parameters = self.flatten_params()


        for e in range(epochs):
            # if X.shape[0] <= 32: # no need for mini batches
            self.weight = []
            self.biases = []
            # result = minimize(self.loss, initial_parameters, args=(X, y), method='L-BFGS-B', options={'disp': False}, jac=self.gradient)
            # initial_parameters = result.x
            result = fmin_l_bfgs_b(self.loss, initial_parameters, args=(X, y), approx_grad=False)
            initial_parameters = result[0]


            # else: # split into miniatches
            #     perm = np.random.permutation(X.shape[0])
            #     X_shuf = X[perm]
            #     y_shuf = y[perm]

            #     batch_size = 32
            #     n_batches = math.ceil(X.shape[0] / batch_size)
            #     batch_size = int(X.shape[0] / n_batches)
            #     # print("n_batches ", n_batches)
            #     for i in range(n_batches):
            #         X_batch = X_shuf[i*batch_size:(i+1)*batch_size]
            #         y_batch = y_shuf[i*batch_size:(i+1)*batch_size]
            #         self.weight = []
            #         self.biases = []
            #         # result = minimize(self.loss, initial_parameters, args=(X_batch, y_batch), method='L-BFGS-B', options={'disp': False}, jac=self.gradient)
            #         # initial_parameters = result.x
            #         result = fmin_l_bfgs_b(self.loss, initial_parameters, fprime=self.gradient, args=(X_batch, y_batch), maxiter=10000, disp=False)
            #         initial_parameters = result[0]

        # print(result)
        self.set_params(result[0])
        # self.set_params(result.x)


    def flatten_params(self):
        return np.concatenate([w.flatten() for w in self.weight] + [b.flatten() for b in self.biases])


    def set_params(self, params):
        shapes = self.weight_shapes + self.biases_shapes
        num_layers = len(self.weight_shapes)


        reshaped = []
        start_idx = 0
        for shape in shapes:
            num_params = np.prod(shape)
            end_idx = start_idx + num_params
            reshaped.append(params[start_idx:end_idx].reshape(shape))
            start_idx = end_idx
        
        self.weight = reshaped[:num_layers]
        self.biases = reshaped[num_layers:]

    def weights(self):
        weights = []
        for i in range (len(self.weight)):
            weights.append(np.vstack((self.biases[i].T, self.weight[i].T)))

        return weights

class ANNClassification(ANN):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_)
        self.encoder = None

    def feedforward(self, x):
        self.hidden_layer_feedforward(x)
        zi = np.dot(self.weight[-1], self.activations[-1]) + self.biases[-1].flatten()
        self.zs.append(zi)
        self.activations.append(softmax(zi))
        
    def loss(self, params, X, Y): # for a batch
        self.weight = []
        self.biases = []

        self.set_params(params)

        loss = 0
        for x, y in zip(X, Y): # for each element in the batch
            self.activations = []
            self.zs = []
            self.feedforward(x)
            loss += - self.log_loss2(y, self.activations[-1]) 


        grad = self.gradient(self.flatten_params(), X, Y)

        loss = loss / X.shape[0]+ self.lambda_ * np.sum([np.sum(w**2) for w in self.weight])

        return loss, grad

    def fit(self, X, y, epochs=1):
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        self.units = [n_features] + self.units + [n_classes]          
        self.initialize_weight()


        # clf = MLPClassifier(hidden_layer_sizes= tuple(self.units), max_iter=10000, alpha=self.lambda_, activation='logistic', solver='lbfgs')
        # clf.fit(X, y)
        # self.weight = clf.coefs_
        # self.biases = clf.intercepts_
        # self.clf = clf
        # print(self.weight)
        # return self
        # print("Before optimization")
        # print(self.weight)
        # print(self.biases)

        self.encoder = OneHotEncoder(handle_unknown='ignore')
        y = self.encoder.fit_transform(y.reshape(-1, 1)).toarray() #target value encoding

        self.optimize(X, y, epochs=epochs)

        # print("After optimization")
        # print(self.weight)
        # print(self.biases)

        # print(self.weight)
        # print(self.biases)
        return self

    def predict(self, X):
        predictions = []
        # p = self.clf.predict_proba(X)

        for x in X:
            self.activations = []
            self.zs = []
            self.feedforward(x)

            predictions.append(list(self.activations[-1]))

        # print(X)
        # print(predictions)
        return np.array(predictions)



class ANNRegression(ANN):

    def feedforward(self, x):
        self.hidden_layer_feedforward(x)
        zi = np.dot(self.weight[-1], self.activations[-1]) + self.biases[-1].flatten()
        self.zs.append(zi)
        self.activations.append(zi) # output je samo linearna kombinacija

    def loss(self, params, X, Y): # MSE
        self.weight = []
        self.biases = []

        self.set_params(params)
        loss = 0
        for x, y in zip(X, Y): # for each element in the batch
            self.activations = []
            self.zs = []
            self.feedforward(x)
            loss += mean_squared_error([y], self.activations[-1])
        loss = loss / X.shape[0]

        grad = self.gradient(self.flatten_params(), X, Y)

        return loss + self.lambda_ * np.sum([np.sum(w**2) for w in self.weight]), grad

    def fit(self, X, y):
        n_features = X.shape[1]
        self.units = [n_features] + self.units + [1]          
        self.initialize_weight()

        self.optimize(X, y)

        return self
    
    def predict(self, X):

        predictions = []
        for x in X:
            self.activations = []
            self.zs = []
            self.feedforward(x)
            predictions.append(self.activations[-1])

        return np.array(predictions).flatten()


def create_final_predictions():


    fitter = ANNClassification(units=[10, 10, 10, 10, 10], lambda_=0.0001)

    df = pd.read_csv("train.csv")
    df = df.drop(columns=["id"])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    names = np.unique(y)
    df = pd.read_csv("test.csv")
    ids = df["id"]
    t = time.time()
    fitter.fit(X, y)
    print("fitting time ", (time.time() - t) /60)
    
    df = pd.read_csv("test.csv")
    ids = df["id"]
    df = df.drop(columns=["id"])
    X = df.values
    t = time.time()
    predictions = fitter.predict(X)
    print("predicting time ", time.time() - t)
    
    df = pd.DataFrame(predictions, columns=names )
    # p = np.vstack((names, p))
    # np.savetxt("final.txt",p , fmt="%s", delimiter=",")
    df.index = ids
    df.to_csv("final.txt", index=True, header=True, sep=",")
    



if __name__ == '__main__':
    x = np.array([[0, 0, 2],
                        [0, 1, 3],
                        [1, 0, 2],
                        [1, 1, 2]])
    y = np.array([0, 1, 1, 0])

    # error_class = []
    # error_reg = []
    # for i in range(100):

    #     encoder = OneHotEncoder(handle_unknown='ignore')
    #     y_ohe = encoder.fit_transform(y.reshape(-1, 1)).toarray() #target value encoding
    #     fitter2 = ANNClassification(units=[3, 10, 10, 10, 2], lambda_=0.0001)
    #     fitter2.initialize_weight()

    #     gradient = fitter2.gradient(fitter2.flatten_params(), x, y_ohe)
    #     numerical_gradient = fitter2.compute_numerical_gradient(x, y_ohe)

    #     diff = np.linalg.norm(gradient - numerical_gradient)
    #     error_class.append(diff)
    #     # print(gradient)
    #     # print(numerical_gradient)
    #     print("Gradient difference for Classification: ", np.linalg.norm(gradient - numerical_gradient))

    #     fitter2 = ANNRegression(units=[3, 10, 10, 10, 1], lambda_=0.0001)
    #     fitter2.initialize_weight()
    #     gradient = fitter2.gradient(fitter2.flatten_params(), x, y)
    #     numerical_gradient = fitter2.compute_numerical_gradient(x, y)

    #     diff = np.linalg.norm(gradient - numerical_gradient)
    #     error_reg.append(diff)
    #     print("Gradient difference for Regression: ", np.linalg.norm(gradient - numerical_gradient))


    # print("average error for Classification: ", np.mean(error_class))
    # print("average error for Regression: ", np.mean(error_reg))




