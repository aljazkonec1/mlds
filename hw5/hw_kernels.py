import numpy as np
import cvxopt
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
import time

class Linear:
    def __init__(self):
        pass

    def __call__(self, A, B):
        return np.dot(A, B.T)

class Polynomial:
    def __init__(self, M):
        self.M = M

    def __call__(self, A, B):
        return ((1 + np.dot(A, B.T)) ** self.M).T

class RBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, A, B):
        a_shape = len(A.shape)
        b_shape = len(B.shape)

        if a_shape == 1 and b_shape == 1: # both are vectors
            dist_sq = A.dot(A.T) - 2 * A.dot(B.T) + B.dot(B.T)
        elif a_shape == 1: # A is a vector, B is a matrix
            dist_sq = A.dot(A.T) - 2 * A.dot(B.T) + np.sum(B * B, axis=1)
        elif b_shape == 1: # A is a matrix, B is a vector
            dist_sq = np.sum(A * A, axis=1) - 2 * A.dot(B.T) + B.dot(B.T)
        else: # both are matrices
            dist_sq = np.sum(np.power(A-B[:, np.newaxis], 2),axis=-1)
        
        return np.exp(-dist_sq / (2 * self.sigma**2))

class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_ = 0.1):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.support_vectors = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        K = self.kernel(X, X)

        self.alpha = np.linalg.inv(K + self.lambda_ * np.eye(len(K))).dot(y)
        sv = np.abs(self.alpha) > 1e-4
        self.support_vectors = X[sv]
        return self

    def predict(self, X):
        K = self.kernel(X, self.X_train)

        return self.alpha.T.dot(K)


    def get_params(self, deep=True):
        return {'kernel': self.kernel, 'lambda_': self.lambda_}
    
    def set_params(self, **parameters):
        for par, val in parameters.items():
            setattr(self, par, val)
        return self
    

class SVR:
    def __init__(self, kernel, epsilon, lambda_ = 0.1):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = 1 / self.lambda_
        self.epsilon = epsilon
        self.alpha = None
        self.alpha_star = None
        self.X_train = None
        self.b = None
        self.alphas_pos = None
        self.alphas_neg = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        K = self.kernel(X, X)
        # print("X", X)
        # print(K)
        # print(K.shape)
        self.y_train = y


        off_diagonal = -K.copy()
        P = np.zeros((2 * n_samples, 2 * n_samples))
        P[::2, ::2] = K   # alpha_i * alpha_j
        P[1::2, 1::2] = K  # alpha*_i * alpha*_j
        P[::2, 1::2] = off_diagonal  # alpha_i * alpha*_j
        P[1::2, ::2] = off_diagonal  # alpha*_i * alpha_j

        P = matrix(P)

        q_pos = self.epsilon - y
        q_neg = self.epsilon + y
        q = np.empty(2 * n_samples)
        q[::2] = q_pos
        q[1::2] = q_neg
        q = matrix(q)

        # 0 <= alpha, alpha* <= C
        G = np.vstack([
            np.eye(2 * n_samples) * -1,  # -alpha_i, -alpha*_i <= 0
            np.eye(2 * n_samples)        #  alpha_i, alpha*_i <= C
        ])
        G = matrix(G)
        
        h = np.hstack([
            np.zeros(2 * n_samples),    # -alpha_i, -alpha*_i <= 0
            np.ones(2 * n_samples) * self.C  # alpha_i, alpha*_i <= C
        ])
        h = matrix(h)

        A = np.empty(2 * n_samples)
        A[::2] = 1
        A[1::2] = -1
        A = matrix(A, (1, 2 * n_samples)) #sum(alpha_i - alpha*_i) = 0 constraint
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])

        self.alphas_pos = alpha[::2]
        self.alphas_neg = alpha[1::2]
        self.alpha = self.alphas_pos - self.alphas_neg
        self.b = np.array(solution['y'])[0]

        sv = np.abs(self.alpha) > 1e-4
        self.support_vectors = X[sv]
        return self
    

    def get_alpha(self):
        return np.array([self.alphas_pos, self.alphas_neg]).T

    def get_b(self):
        return self.b
    
    def predict(self, X):
        K = self.kernel(X, self.X_train)
        return (np.dot(self.alpha.T, K) + self.b).flatten()

    def get_params(self, deep=True):
        return {'kernel': self.kernel, 'epsilon': self.epsilon, 'lambda_': self.lambda_}
    
    def set_params(self, **parameters):
        for par, val in parameters.items():
            setattr(self, par, val)
        return self

if __name__ == "__main__":
    # X = np.array([[1, 2], [3, 4], [5, 6]])
    # y = np.array([0., 1, 2, 3])
    # x = np.array([0.5, 1.])
    # kernel = RBF(sigma= 1)

    
    # print(kernel(X, X+0.5))
    # print(kernel(X+0.5, X))

    X = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
    y = np.array([0, 0, 1, 1])
    # fitter = SVR(kernel=RBF(sigma=0.5), lambda_=0.001, epsilon=0.1)
    
    # m = fitter.fit(X, y)
    # pred = m.predict(X) 

    # print(pred)
    fitter = SVR(kernel=Polynomial(M=2), lambda_=0.0001, epsilon=0)
    
    m = fitter.fit(X, y)
    pred = m.predict(X) 

    print(pred)

    # fitter = SVR(kernel=Linear(), lambda_=0.001, epsilon=0.1)
    
    # m = fitter.fit(X, y)
    # pred = m.predict(X) 

    # print(pred)

    sine = pd.read_csv('sine.csv')
    X = sine['x'].values
    y = sine['y'].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    X_train = X
    y_train = y
    X_test = X
    y_test = y

    X_train = X_train[:, np.newaxis]
    X_test = X_test[:, np.newaxis]

    sigmas = [0.1, 0.5, 1, 2, 3, 4, 5, 10]
    Ms = [1, 2, 3, 4]
    lambdas = [0.0001, 0.001, 0.01, 0.1, 1]
    epsilons = [0, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    res = {}
    for sigma in sigmas:
        for l in lambdas:
            fitter = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=l)
            fitter.fit(X_train, y_train)
            y_pred = fitter.predict(X_test)
            mse = np.sqrt(mean_squared_error(y_test, y_pred))
            res[f'{sigma}-{l}'] = mse


    df = pd.DataFrame(res.items(), columns=['sigma-lambda', 'mse'])
    # df.to_csv('sine_KRR_rbf.csv', index=False)

    res = {}
    for m in Ms:
        for l in lambdas:
            for e in epsilons:
                fitter = SVR(kernel=Polynomial(M=m), lambda_=l, epsilon=e)
                print(f'{m}-{l}-{e}')
                fitter.fit(X_train, y_train)
                y_pred = fitter.predict(X_test)
                mse = np.sqrt(mean_squared_error(y_test, y_pred))
                res[f'{m}-{l}-{e}'] = mse


    df = pd.DataFrame(res.items(), columns=['M-lambda-e', 'rmse'])
    print(df)
    # df.to_csv('sine_SVR_polynomial.csv', index=False)

    res = {}
    for s in sigmas:
        for l in lambdas:
            for e in epsilons:
                fitter = SVR(kernel=RBF(sigma=s), lambda_=l, epsilon=e)
                print(f'{s}-{l}-{e}')
                fitter.fit(X_train, y_train)
                y_pred = fitter.predict(X_test)
                mse = np.sqrt(mean_squared_error(y_test, y_pred))
                res[f'{s}-{l}-{e}'] = mse


    df = pd.DataFrame(res.items(), columns=['M-lambda', 'rmse'])
    print(df)
    # df.to_csv('sine_SVR_RBF.csv', index=False)

    res = {}
    for m in Ms:
        for l in lambdas:
            fitter = KernelizedRidgeRegression(kernel=Polynomial(M=m), lambda_=l)
            fitter.fit(X_train, y_train)
            y_pred = fitter.predict(X_test)
            mse = np.sqrt(mean_squared_error(y_test, y_pred))
            res[f'{m}-{l}'] = mse


    df = pd.DataFrame(res.items(), columns=['M-lambda', 'rmse'])
    # df.to_csv('sine_KRR_polynomial.csv', index=False)

    svr_rbf = SVR(kernel=RBF(sigma=1), lambda_=1.5, epsilon=0.7)
    svr_poly = SVR(kernel=Polynomial(M=4), lambda_=0.0001, epsilon=2)
    krr_rbf = KernelizedRidgeRegression(kernel=RBF(sigma=2), lambda_=0.1)
    krr_poly = KernelizedRidgeRegression(kernel=Polynomial(M=4), lambda_=0.1)

    svr_rbf.fit(X_train, y_train)
    svr_poly.fit(X_train, y_train)
    krr_rbf.fit(X_train, y_train)
    krr_poly.fit(X_train, y_train)


    y_pred_svr_rbf = svr_rbf.predict(X_test)
    y_pred_svr_poly = svr_poly.predict(X_test)
    y_pred_krr_rbf = krr_rbf.predict(X_test)
    y_pred_krr_poly = krr_poly.predict(X_test)

    print("nr_support_vectors_svr_rbf", len(svr_rbf.support_vectors))
    print("nr_support_vectors_svr_poly", len(svr_poly.support_vectors))

    figure = plt.figure(figsize=(10, 5))
    plt.scatter(X_test, y_test)
    plt.scatter(X_test, y_pred_svr_rbf, label='SVR RBF sigma=1 lamda=1.5, epsilon=0.7', color='red', s=10)
    plt.scatter(X_test, y_pred_svr_poly, label='SVR Poly M=4, lambda=0.0001, epsilon=2', color='blue', s=10)
    plt.scatter(X_test, y_pred_krr_rbf, label='KRR RBF sigma=2, lambda=0.1', color='green', s=10)
    plt.scatter(X_test, y_pred_krr_poly, label='KRR Poly M=4, lambda=0.1', color='black', s=10)
    plt.legend()
    plt.savefig('report/figures/sine.pdf')

    housing2r = pd.read_csv('housing2r.csv')

    X_train = housing2r.drop('y', axis=1).iloc[:160, :]

    x_mean = X_train.mean()
    x_std = X_train.std()
    X_train = (X_train - x_mean) / x_std
    X_train = X_train.values

    y_train = housing2r['y'].values[:160]
    X_test = housing2r.drop('y', axis=1).iloc[160:, :]
    X_test = (X_test - x_mean) / x_std
    X_test = X_test.values

    y_test = housing2r['y'].values[160:]

    sigmas = [0.1, 1, 2, 3, 4, 5]
    Ms = [1, 2, 3, 4, 5]


    mses_RBF = []
    for sigma in sigmas:
        fitter = KernelizedRidgeRegression(kernel=RBF(sigma=sigma), lambda_=1)
        fitter.fit(X_train, y_train)
        y_pred = fitter.predict(X_test)
        mse1 = np.sqrt(mean_squared_error(y_test, y_pred))

        fitter = SVR(kernel=RBF(sigma=sigma), lambda_=1, epsilon=8)
        fitter.fit(X_train, y_train)
        y_pred = fitter.predict(X_test)
        mse2 = np.sqrt(mean_squared_error(y_test, y_pred))
        nr_supports = len(fitter.support_vectors)
        
        mses_RBF.append([sigma, mse1, mse2, nr_supports])

    mses_poly = []
    for m in Ms:
        fitter = KernelizedRidgeRegression(kernel=Polynomial(M=m), lambda_=1)
        fitter.fit(X_train, y_train)
        y_pred = fitter.predict(X_test)
        mse1 = np.sqrt(mean_squared_error(y_test, y_pred))

        fitter = SVR(kernel=Polynomial(M=m), lambda_=1, epsilon=8)
        fitter.fit(X_train, y_train)
        y_pred = fitter.predict(X_test)
        mse2 = np.sqrt(mean_squared_error(y_test, y_pred))
        nr_supports = len(fitter.support_vectors)
        mses_poly.append([m, mse1, mse2, nr_supports])

    sigmas = [0.1, 1, 2, 3, 4, 5]
    Ms = [1, 2, 3, 4, 5]

    t = time.time()
    params = {'lambda_': np.array([0.0001, 0.001, 0.01, 1, 2, 3, 4, 5, 10])}
    mses_RBF_cv = []
    for sigma in sigmas:
        
        model = GridSearchCV(KernelizedRidgeRegression(kernel=RBF(sigma=sigma)), params, cv=5, scoring='neg_mean_squared_error')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse1 = np.sqrt(mean_squared_error(y_test, y_pred))

        fitter = GridSearchCV(SVR(kernel=RBF(sigma=sigma), epsilon=8), params, cv=5, scoring='neg_mean_squared_error')
        fitter.fit(X_train, y_train)
        y_pred = fitter.predict(X_test)
        print("nr_supports: ", len(fitter.best_estimator_.support_vectors))
        nr_supports = len(fitter.best_estimator_.support_vectors)
        mse2 = np.sqrt(mean_squared_error(y_test, y_pred))
        
        mses_RBF_cv.append([sigma, mse1, mse2, nr_supports])

    print(mses_RBF_cv)
    mses_poly_cv = []
    for m in Ms:
        fitter = GridSearchCV(KernelizedRidgeRegression(kernel=Polynomial(M=m)), params, cv=5, scoring='neg_mean_squared_error')
        fitter.fit(X_train, y_train)
        y_pred = fitter.predict(X_test)
        mse1 = np.sqrt(mean_squared_error(y_test, y_pred))
        
        fitter = GridSearchCV(SVR(kernel=Polynomial(M=m), epsilon=8), params, cv=5,scoring='neg_mean_squared_error')
        fitter.fit(X_train, y_train)
        y_pred= fitter.predict(X_test)
        nr_supports = len(fitter.best_estimator_.support_vectors)
        mse2 = np.sqrt(mean_squared_error(y_test, y_pred))
        
        mses_poly_cv.append([m, mse1, mse2, nr_supports])
    print(mses_poly_cv)
    print('Time elapsed CV:', int((time.time() - t)))

    figure, ax = plt.subplots(2, 2, figsize=(20, 13))
    ax[0, 0].plot([x[0] for x in mses_RBF], [x[1] for x in mses_RBF], label='KRR RBF lambda=1')
    ax[0, 1].plot([x[0] for x in mses_RBF], [x[2] for x in mses_RBF], label='SVR RBF lambda=1')
    ax[0,0].plot([x[0] for x in mses_RBF_cv], [x[1] for x in mses_RBF_cv], label='KRR RBF CV')
    ax[0,1].plot([x[0] for x in mses_RBF_cv], [x[2] for x in mses_RBF_cv], label='SVR RBF CV')

    ax[1, 0].plot([x[0] for x in mses_poly], [x[1] for x in mses_poly], label='KRR Poly lambda=1')
    ax[1, 1].plot([x[0] for x in mses_poly], [x[2] for x in mses_poly], label='SVR Poly lambda=1')
    ax[1, 0].plot([x[0] for x in mses_poly_cv], [x[1] for x in mses_poly_cv], label='KRR Poly CV')
    ax[1, 1].plot([x[0] for x in mses_poly_cv], [x[2] for x in mses_poly_cv], label='SVR Poly CV')


    for r in mses_RBF:
        ax[0, 1].annotate(str(r[3]), (r[0], r[2]))

    for r in mses_poly:
        ax[1, 1].annotate(str(r[3]), (r[0], r[2]))

    for r in mses_RBF_cv:
        ax[0, 1].annotate(str(r[3]), (r[0], r[2]))

    for r in mses_poly_cv:
        ax[1, 1].annotate(str(r[3]), (r[0], r[2]))


    ax[0,1].set_title('Support Vector Regression with RBF Kernel')
    ax[0, 0].set_title('Kernelized Ridge Regression with RBF Kernel')
    ax[1, 0].set_title('Kernelized Ridge Regression with Polynomial Kernel')
    ax[1, 1].set_title('Support Vector Regression with Polynomial Kernel')

    ax[0,0].set_ylabel('RMSE')
    ax[1,0].set_ylabel('RMSE')
    ax[0,0].set_xlabel('Sigma')
    ax[0,1].set_xlabel('Sigma')
    ax[1,0].set_xlabel('M')
    ax[1,1].set_xlabel('M')
    ax[0,0].legend()
    ax[0,1].legend()
    ax[1,0].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.savefig('report/figures/housing_results.pdf')
