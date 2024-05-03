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

from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
        self.biases = [np.random.randn(1, y) for y in self.units[1:]]
        self.weight_shapes = [(x, y) for x, y in zip(self.units[:-1], self.units[1:])]
        self.biases_shapes = [( 1, y) for y in self.units[1:]]


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

        
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def cost_derivative(self, output_activations, y):
        return output_activations - y
    
    def hidden_layer_feedforward(self, x): # feed forward one element, do not compute output layer
        self.activations.append(x)

        for i in range(len(self.weight)-1):
            zi = np.dot(self.activations[i], self.weight[i]) + self.biases[i] 
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

    def gradient(self, params, X, y): # backpropagation
        self.set_params(params)

        nabla_b = [np.zeros(b).T for b in self.biases_shapes]
        nabla_w = [np.zeros(w) for w in self.weight_shapes]

        delta = self.cost_derivative(self.activations[-1], y) 
        # * self.sigmoid_derivative(self.zs[-1]) #BP1

        nabla_b[-1] = np.mean(delta, axis=0, keepdims=True) #BP3
        nabla_w[-1] = np.dot(delta.T, self.activations[-2]).T #BP4

        for l in range(2, len(self.units)):
            delta = np.dot( delta, self.weight[-l+1].T) * self.sigmoid_derivative(self.zs[-l]) #BP2
            nabla_b[-l] = np.mean(delta, axis=0, keepdims=True)
            nabla_w[-l] =  np.dot(delta.T, self.activations[-l-1]).T
        

        nabla_w = [nw/X.shape[0] + 2 * self.lambda_ * w for nw, w in zip(nabla_w, self.weight)]

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
            result = fmin_l_bfgs_b(self.loss, initial_parameters, args=(X, y), approx_grad=False)
            initial_parameters = result[0]

        self.set_params(result[0])


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
            # print(self.biases[i].T)
            # print(self.weight[i].T)
            weights.append(np.vstack((self.biases[i], self.weight[i])))

        return weights

class ANNClassification(ANN):
    def __init__(self, units, lambda_):
        super().__init__(units, lambda_)
        self.encoder = None

    def feedforward(self, x):
        self.activations = []
        self.zs = []
        self.hidden_layer_feedforward(x)
        zi = np.dot(self.activations[-1], self.weight[-1]) + self.biases[-1]

        self.zs.append(zi)
        self.activations.append(softmax(zi, axis= 1))
        
    def loss(self, params, X, Y): # for a batch
        self.weight = []
        self.biases = []
        self.set_params(params)


        self.feedforward(X)
        loss= log_loss(Y, self.activations[-1]) 
        grad = self.gradient(self.flatten_params(), X, Y)

        loss += self.lambda_ * np.sum([np.sum(w**2) for w in self.weight])

        return loss, grad

    def fit(self, X, y, epochs=1):
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        self.units = [n_features] + self.units + [n_classes]          
        self.initialize_weight(s=0)

        self.encoder = OneHotEncoder(handle_unknown='ignore')
        y = self.encoder.fit_transform(y.reshape(-1, 1)).toarray() #target value encoding
        self.optimize(X, y, epochs=epochs)

        return self

    def predict(self, X):
        self.feedforward(X)
        return self.activations[-1]



class ANNRegression(ANN):

    def feedforward(self, x):
        self.hidden_layer_feedforward(x)
        zi = np.dot( self.activations[-1], self.weight[-1]) + self.biases[-1]
        self.zs.append(zi)
        self.activations.append(zi) # output je samo linearna kombinacija

    def loss(self, params, X, Y): # MSE
        self.weight = []
        self.biases = []

        self.set_params(params)
        self.activations = []
        self.zs = []
        self.feedforward(X) 
        loss = mean_squared_error(Y, self.activations[-1])
        loss = loss / X.shape[0]

        grad = self.gradient(self.flatten_params(), X, Y)

        return loss + self.lambda_ * np.sum([np.sum(w**2) for w in self.weight]), grad

    def fit(self, X, y):
        n_features = X.shape[1]
        self.units = [n_features] + self.units + [1]          
        self.initialize_weight()    
        y = y.reshape(-1, 1)
        self.optimize(X, y)

        return self
    
    def predict(self, X):

        self.feedforward(X)
        return self.activations[-1].flatten()



def create_final_predictions():

    df = pd.read_csv("train.csv")
    df = df.drop(columns=["id"])

    X_train = df.drop(columns='target')
    y_train = df['target']

    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train = (X_train - train_mean) / train_std

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    names = np.unique(y_train)


    t = time.time()
    fitter = ANNClassification(units=[128], lambda_=0.0001)
    m = fitter.fit(X_train, y_train, epochs=1)
    ttime = time.time() - t
    print("Training time: ", time.time() - t)

    df = pd.read_csv("test.csv")
    ids = df["id"]
    df = df.drop(columns=["id"])
    df = (df - train_mean) / train_std #normalizacija
    df = df.to_numpy()
    print(df.shape)
    t = time.time()
    predictions = m.predict(df)
    print("predicting time ", time.time() - t)
    
    df = pd.DataFrame(predictions, columns=names )
    df.index = ids
    df.to_csv("final.txt", index=True, header=True, sep=",")
    


if __name__ == '__main__':
    x = np.array([[0, 0, 2],
                        [0, 1, 3],
                        [1, 0, 2],
                        [1, 1, 2]])
    y = np.array([0, 1, 1, 0])
    


    # fitter = ANNRegression(units=[5, 3], lambda_=0.0001)
    # m = fitter.fit(x, y)
    # print(m.predict(x))
    # print(m.weights())

    # print("Creating final predictions")
    # # create_final_predictions()    

    # ## test models with diffrent parameters
    # df = pd.read_csv("train.csv")
    # df = df.drop(columns=["id"])


    # # params = [ [128], [128, 128], [64], [64, 64], [128, 2, 2, 2, 2, 2, 2]]
    # param_names = ["128", "128-128", "64", "64-64", "128-2-2-2-2-2-2"]
    # params = [ [128]]

    # for i, param in enumerate(params):


    #     X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), df['target'], train_size=0.8)

    #     train_mean = X_train.mean()
    #     train_std = X_train.std()

    #     X_train = (X_train - train_mean) / train_std
    #     X_test = (X_test - train_mean) / train_std
    #     X_train = X_train.to_numpy()
    #     X_test = X_test.to_numpy()

    #     y_train = y_train.to_numpy()
    #     y_test = y_test.to_numpy()
    #     names = np.unique(y_train)

    #     t = time.time()
    #     fitter = ANNClassification(units=param, lambda_=0.0001)
    #     m = fitter.fit(X_train, y_train, epochs=1)
    #     ttime = time.time() - t
    #     print("Training time: ", time.time() - t)

    #     t = time.time()
    #     predictions = m.predict(X_test)
    #     print("predicting time ", time.time() - t)

    #     y_test_onehot = m.encoder.transform(y_test.reshape(-1, 1)).toarray()

    #     # print(predictions)
    #     logloss = log_loss(y_test_onehot, predictions)
    #     print("Logloss: ", logloss)

    #     name = "final/" + param_names[i] + ".txt"
    #     np.savetxt(name, [logloss, ttime], delimiter=",")
    #     np.savetxt("final/predictions" + param_names[i] + ".txt", predictions, delimiter=",")
    #     np.savetxt("final/y_test" + param_names[i] +".txt", y_test_onehot, delimiter=",")





