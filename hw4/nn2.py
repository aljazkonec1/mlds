import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def sigmoid_derivative(x):
#     return x * (1-x)

# def sigmoid_derivative(x):
#     return sigmoid(x) * (1 - sigmoid(x))

def softmax(z): ### IZ HW3 !!
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # odstejemo zadnji stolpec za referenco z[:, -1].reshape(-1, 1) ALI np.max(z, axis=1, keepdims=True)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def log_loss(y, preds): ### IZ HW3 !!
    return (1/len(y)) * np.sum(np.log(preds[np.arange(len(y)), y]))

def mse(y, preds):
    return (1/(2 * len(y))) * np.sum((y-preds)**2)

def numerical_gradient(f, w, h = None):

    if h is None:
        h = 0.001

    n = len(w)
    grads = np.zeros(n)

    for i in range(n):
        w_plus = w.copy()
        w_plus[i] += h

        w_minus = w.copy()
        w_minus[i] -= h

        grads[i] = (f(w_plus) - f(w_minus)) / (2 * h)

    return grads

def one_hot(y):
    vals = np.unique(y)

    ohe = np.zeros((len(y), len(vals)))

    for i, val in enumerate(vals):
        ohe[:, i] = (y == val).astype(int)

    return ohe

class ANNClassification:
    def __init__(self, units, lambda_):
        self.hidden_sizes = units
        self.lambda_ = lambda_
        
    def fit(self, X, y):
        m = Model(X, y, 'C', self.hidden_sizes, self.lambda_).train(X, y)
        return m

class ANNRegression:
    def __init__(self, units, lambda_):
        self.hidden_sizes = units
        self.lambda_ = lambda_
        
    def fit(self, X, y):
        m = Model(X, y, 'R', self.hidden_sizes, self.lambda_).train(X,y)
        return m
    
class Model:
    def __init__(self, X, y, type, hidden_sizes, lambda_):
        self.X = X
        self.y = y
        self.type = type # 'R' regresija, 'C' klasifikacija
        self.hidden_sizes = hidden_sizes
        self.lambda_ = lambda_

        self.input_size = X.shape[1]
        if self.type == 'R':
            self.output_size = 1
        else:
            self.output_size = len(np.unique(y))

        # np.random.seed()

        self.Ws = [np.random.randn(prev, cur) for prev, cur in zip([self.input_size] + self.hidden_sizes, self.hidden_sizes + [self.output_size])]
        self.Bs = [np.random.randn(1, cur) for cur in hidden_sizes + [self.output_size]]

    def forward(self, X):
        As = [X]

        for i in range(len(self.hidden_sizes)):
            z = np.dot(As[i], self.Ws[i]) + self.Bs[i]
            As.append(sigmoid(z))
            # print(i)

        if self.type == 'C': # ce klasifikacija, na koncu še sigmoid in softmax
            z = np.dot(As[-1], self.Ws[-1]) + self.Bs[-1]
            As.append(softmax(z))

        else:
            z = np.dot(As[-1], self.Ws[-1]) + self.Bs[-1]
            As.append(z)

        return As
    
    def backward(self, y, As):
        grads_w = [np.zeros_like(w) for w in self.Ws]
        grads_b = [np.zeros_like(b) for b in self.Bs]
 
        pred = As[-1]

        if self.type == 'C':
            y = one_hot(y)
        else:
            y = y.reshape(-1,1)

        delta = (pred - y)

        for i in reversed(range(len(self.Ws))):
            grads_w[i] = np.dot(As[i].T, delta) * (1/len(pred)) + 2 * self.lambda_ * self.Ws[i]
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)
            delta = np.dot(delta, self.Ws[i].T) * ((As[i]) * (1-As[i]))

        grads = self.grads_to_1D(grads_w, grads_b)

        return grads
    
    def numerical_grad(self, params, X, y, split):
        h = 1e-10

        grads = [np.zeros(param.shape) for param in params]
        
        for i in range(len(grads)):
            params_plus = params.copy()
            params_plus[i] += h

            params_minus = params.copy()
            params_minus[i] -= h

            grads[i] = (self.loss_params(params_plus, X, y, split) - self.loss_params(params_minus, X, y, split)) / (2 * h)

        return grads

    def predict(self, X):
        res = (self.forward(X))[-1]

        if self.type == 'C':
            return res

        return res.reshape(-1)
    
    def loss(self, X, y):
        preds = self.predict(X).reshape(-1)
        
        if self.type == 'C':
            res = log_loss(y, preds)
            # print(res)
            return -res
        else: 
            res = mse(y, preds)
            # print(res)
            return res
    
    def loss_As(self, As, y):
        preds = As[-1]
        
        if self.type == 'C':
            res = log_loss(y, preds)
            # print(res)
            return -res
        else: 
            preds = preds.reshape(-1) # flattamo, ko regresija
            res = mse(y, preds)
            # print(res)
            return res
        
    def loss_params(self, params, X, y, split):
        penalty = self.lambda_ * np.sum((params[:split])**2)

        Ws, Bs = self.params_to_normal(params, split)

        As = [X]

        for i in range(len(self.hidden_sizes)):
            z = np.dot(As[i], Ws[i]) + Bs[i]
            As.append(sigmoid(z))

        if self.type == 'C': # ce klasifikacija, na koncu še softmax
            z = np.dot(As[-1], Ws[-1]) + Bs[-1]
            return (- log_loss(y, softmax(z))) + penalty

        else:
            z = np.dot(As[-1], Ws[-1]) + Bs[-1]
            return mse(y, z.reshape(-1)) + penalty

    def fun_to_optimise(self, params, X, y, split):

        penalty = self.lambda_ * np.sum((params[:split])**2)# lazje ga je poracunat, dokler so vsi W eden za drugim

        Ws, Bs = self.params_to_normal(params, split)
        self.Ws = Ws
        self.Bs = Bs

        As = self.forward(X)
        loss = self.loss_As(As,y)
        grads = self.backward(y=y, As=As)
        # num_grads = self.numerical_grad(params,X,y,split)
        # print('--------------------------------------------------------------')
        # print(np.round(grads,3))
        # print(np.round(num_grads, 3))

        # print(loss)

        return loss + penalty, grads

    def train(self, X, y):
        params, split = self.params_to_1D()
        
        result = fmin_l_bfgs_b(self.fun_to_optimise, params, args=(X, y, split), approx_grad=False)

        self.Ws, self.Bs = self.params_to_normal(result[0], split) 

        return self

    def params_to_1D(self):
        params = list(self.Ws[0].flatten())

        for w in self.Ws[1:]:
            params += list(w.flatten())

        split = len(params)

        for b in self.Bs:
            params += list(b.flatten())

        return params, split
    
    def grads_to_1D(self, Ws, Bs):
        grads = list(Ws[0].flatten())

        for w in Ws[1:]:
            grads += list(w.flatten())
        
        for b in Bs:
            grads += list(b.flatten())

        return grads
    
    def params_to_normal(self, params, split):

        Ws_flat = np.array(params[:split])
        Bs_flat = np.array(params[split:])

        Ws = []
        cur = 0
        for w in self.Ws:
            w_size = w.shape[0] * w.shape[1]
            Ws.append(Ws_flat[cur:cur+w_size].reshape(w.shape))
            cur += w_size

        Bs = []
        cur = 0
        for b in self.Bs:
            b_size = b.shape[1]
            Bs.append(Bs_flat[cur:cur+b_size].reshape(b.shape))
            cur += b_size

        return Ws, Bs

    def weights(self):

        # print(self.Bs)
        # print(self.Ws)

        # print(np.vstack((self.Bs, self.Ws)))
        return np.vstack((self.Bs, self.Ws))

def smallExample():
    X = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 0.5],
                    [1, 1],
                    [1, 1.5],
                    [1.5, 1],
                    [2, 2],
                    [2.5, 3]])
    y = np.array([0, 0, 1, 3, 1, 2, 4, 3, 3]) # 1 tam kjer prva 3, 2 namesto 4

    X_2 = np.array([[0, 0],
                    [1, 0.5],
                    [2.5, 3],
                    [5, 2]])
    
    # y_2 = np.array([0, 1, 1, 2])
    
    # l = ANNClassification([5,5], lambda_=0.1).fit(X,y)
    # # print(np.round(l.predict(X), 2))
    # print(np.round(l.predict(X_2), 2))
    
    l = ANNRegression([10,5], lambda_=0.1).fit(X,y)
    # print(np.round(l.predict(X_2), 2))

def hardExample():
    X = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    y = np.array([0, 1, 2, 3])
    hard_y = np.array([0, 1, 1, 0])

    # l = ANNRegression([13, 6], lambda_=0.1).fit(X,hard_y)
    # print(np.round(l.predict(X), 2))

    l = ANNClassification([5], lambda_ = 0.1).fit(X, hard_y)
    # print(np.round(l.predict(X), 2))

if __name__ == '__main__':
    # smallExample()
    hardExample()