import numpy as np
from collections import Counter
import random
import csv
import time
import matplotlib.pyplot as plt

def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    p = int(np.sqrt(X.shape[1]))

    return rand.sample(range(X.shape[1]), p)


class Tree:

    def __init__(self, rand=random.Random(),
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def build(self, X, y):
        cols = self.get_candidate_columns(X, self.rand)

        if (len(y) < self.min_samples) or self.gini_impurity(y) == 0: #create a leaf
            return TreeNode(X, y, columns=cols)

        # print("to je X: ", X)
        # print("to je y: ", y)
        # print("unique X: ", np.unique(X).shape)

        dL, yL, dR, yR, criteria, idx = self.split(X, y, cols)
        node = TreeNode(X, y,columns=cols, criteria=criteria, idx=idx)
        node.L = self.build(dL, yL)
        node.R = self.build(dR, yR)

        return node

    def split(self, X, y, candidate_columns):
        best_idx = -1
        best_gain = -1
        best_criteria = -1

        current_imputiry = self.gini_impurity(y)
        l = len(y)

        for idx in candidate_columns:
            for t in np.unique(X[:, idx]):
                # print(idx, t)

                split = X[:, idx] < t
                dL = X[split, :]
                yL = y[split]

                dR = X[~split, :]
                yR = y[~split]


                l_impurity = self.gini_impurity(yL)
                r_impurity = self.gini_impurity(yR)
                
                quality_of_split = (len(yL) / l) * l_impurity + (len(yR) / l) * r_impurity
                gini_gain = current_imputiry - quality_of_split

                if gini_gain > best_gain:
                
                    # print(gini_gain)
                    best_gain = gini_gain
                    best_idx = idx
                    best_criteria = t
                    best_dL, best_yL, best_dR, best_yR = dL, yL, dR, yR



        return best_dL, best_yL, best_dR, best_yR, best_criteria, best_idx

    def gini_impurity(self, y):
        g = 0
        for k in np.unique(y):
            p = np.sum(y == k) / len(y)
            g += p**2

        return 1 - g
    
class TreeNode:

    def __init__(self, X = None, y = None, columns=None, criteria = None, idx = None, L = None, R = None):
        self.X = X
        self.y = y
        self.columns = columns
        self.criteria = criteria
        self.idx = idx
        self.value = None

        if len(y) > 0:
            # print("to bi moral biti y v tree node: ", y)

            self.value = Counter(y).most_common(1)[0][0]

        # print("to je value ", self.value)
        self.L = L
        self.R = R


    def predict_on_one(self, x):
        if self.L == None:
            return self.value
        
        if x[self.idx] < self.criteria:
            return self.L.predict_on_one(x)
        else:
            return self.R.predict_on_one(x)

    def predict(self, X):
        y = []
        for row in X:
            y.append(self.predict_on_one(row))

        return np.array(y)
    
    def print_tree(self, X, depth=0):
        # print(self.criteria, self.idx, self.value)
        # print(self.L, self.R)
        if(self.criteria == None):
            print("   " * depth + "|—> LEAF:  " + str(len(X)) + " prediction  " + str(self.value))
            return
        indices = X[:, self.idx] <= self.criteria
        X_l = X[indices]
        X_r = X[~indices]

        print("  " * depth + "|—>  t: " + f"{self.criteria:.4}" + "  f_idx: " + str(self.idx) + "  nr_rows_left: " +  str(np.sum(indices)) + "  nr_rows_right: " +  str(np.sum(~indices)))
        if self.L:
            self.L.print_tree(X_l, depth + 1)
        if self.R:
            self.R.print_tree(X_r, depth + 1)

class RandomForest:

    def __init__(self, rand=random.Random(), n=50):
        self.n = n
        self.rand = rand
        self.t = Tree(rand=self.rand, get_candidate_columns=random_sqrt_columns)

    def build(self, X, y):
        trees = []
        oobs = []
        for _ in range(self.n) :


            bootstrap_indices = self.rand.choices(range(len(y)), k=len(y))

            X_b = X[bootstrap_indices]
            y_b = y[bootstrap_indices]
            # print("unique bootstrap indices: ", np.unique(bootstrap_indices))
            # print("len unique bootstrap indices: ", len(np.unique(bootstrap_indices)))

            # print("X_b: ", X_b)
            # print("y_b: ", y_b)
            # print("unique X_b: ", np.unique(X_b).shape)
            p = self.t.build(X_b, y_b)
            # p.print_tree(X_b)

            trees.append(p)
            oobs.append(bootstrap_indices)

        return RFModel(rand=self.rand, trees=trees, oobs=oobs, X=X, y=y )


class RFModel:

    def __init__(self, rand = random.Random(), trees= None, oobs=None, X = None, y = None,):
        self.X = X
        self.y = y
        self.trees = trees
        self.rand = rand
        self.oobs = oobs

    def predict(self, X):
        y = []

        for tree in self.trees:
            y.append(tree.predict(X))

        y = np.array(y).T
        # print("y v rf: ", y)
        # print(y.shape)
        y = np.array([Counter(row).most_common(1)[0][0] for row in y])
        # print("y v rf: ", y)
        
        return y

    def importance(self):

        imps = np.empty((len(self.trees), self.X.shape[1]))
        imps[:, :] = np.nan

        for i, tree in enumerate(self.trees):
            oob = np.array(range(len(self.y)))[np.isin(range(len(self.y)), self.oobs[i], invert=True)]
            
            y_pred = tree.predict(self.X[oob])

            e_orig = self.calculate_accuracy(self.y[oob], y_pred) # original error

            for f in range(self.X.shape[1]):
                X_perm = self.X.copy()
                self.rand.shuffle(X_perm[:, f])
                y_pred_perm = tree.predict(X_perm[oob])
                e_perm = self.calculate_accuracy(self.y[oob], y_pred_perm)
                imps[i, f] = e_perm - e_orig

        return np.nanmean(imps, axis=0)

    def calculate_accuracy(self, y, y_pred): # missclassification rate

        return np.sum(y_pred != y) / len(y)
    

def hw_tree_full(learn, test):
    t = Tree(rand=random.Random())
    p = t.build(learn[0], learn[1])
    learn_pred = p.predict(learn[0])
    
    learn_miss, learn_un = calc_ms_un(learn[1], learn_pred)

    # p = t.build(test[0], test[1])
    test_pred = p.predict(test[0])
    # print(test[0].shape)
    test_miss, test_un = calc_ms_un(test[1], test_pred)
    # p.print_tree(test[0])

    return (learn_miss, learn_un), (test_miss, test_un)

def calc_ms_un(y, y_pred):
    ms = np.sum(y != y_pred) / len(y) # missclassification rate
    un = np.sqrt(np.var(y!=y_pred) / len(y)) # uncertainty
    return ms, un


def hw_randomforests(learn, test, n = 100, rand=random.Random()):
    rf = RandomForest(rand=rand, n=n)
    # print(learn[0].shape)

    p = rf.build(learn[0], learn[1])

    learn_pred = p.predict(learn[0])

    # print(learn_pred)
    # print(learn[1])

    learn_miss, learn_un = calc_ms_un(learn[1], learn_pred)

    # p = rf.build(test[0], test[1])
    # print(test[0].shape)
    # print(test[0])
    test_pred = p.predict(test[0])
    
    test_miss, test_un = calc_ms_un(test[1], test_pred)

    return (learn_miss, learn_un), (test_miss, test_un)

def tki():
    #load from csv
    # df = np.loadtxt('hw1/tki-resistance.csv', delimiter=',')
    with open('hw1/tki-resistance.csv', 'r') as file:
        reader = csv.reader(file)
        legend = next(reader)
        df = list(reader)
        df = np.array(df)
    # print(df)
    learn= df[0:129, :]
    test = df[130:, :]
    learn_X = np.array(learn[:, :-1], dtype=float)

    _, learn_y = np.unique(learn[:, -1], return_inverse=True)
    test_X = np.array(test[:, :-1], dtype=float)
    _, test_y = np.unique(test[:, -1], return_inverse=True)

    return (learn_X, learn_y), (test_X, test_y), np.array(legend)[:-1]

if __name__ == "__main__":
    x = 0
    learn, test, legend = tki()

    print(learn[0].shape)
    start = time.time()
    print("full", hw_tree_full(learn, test))
    print(f"Tree built in: {time.time() - start:.3f}")
    
    print("random forests", hw_randomforests(learn, test, rand= random.Random(0)))
    
    # ns = [10, 20, 50, 100, 200, 300, 400, 500]

    # missclassification = []
    # std_error = []
    # for n in ns:
    #     ms = hw_randomforests(learn, test, n=n,rand=random.Random(0))[1]
    #     missclassification.append(ms[0])
    #     std_error.append(ms[1])
    
    # print(missclassification)
    # ns = [str(n) for n in ns ]
    # plt.bar(ns, missclassification)
    # plt.errorbar(ns, missclassification, yerr=std_error, color='orange', ls='none')

    # plt.plot(ns, missclassification)

    # plt.show()


    # rf = RandomForest(rand=random.Random(0), n=10)
    # p = rf.build(learn[0], learn[1])
    # importance = p.importance(learn[0], learn[1])

    # print(importance)
    
    # y_pred = p.predict(learn[0])
    # print(y_pred)
    # print(learn[1])
    # missclsification= p.calculate_accuracy(learn[1], y_pred)
    # print(missclsification)

    # start = time.time()
    # rf = RandomForest(rand=random.Random(0), n=100)
    # p = rf.build(learn[0], learn[1])
    # print(f"RF model built in: {time.time() - start:.3f}")

    # importance = p.importance(learn[0], learn[1])
    # # top_5_idx = np.argsort(importance)[::-1][:10]
    # print(importance)
    # print(importance.shape)
    # print(importance[top_5_idx])
    # print(legend[top_5_idx])
    # print(top_5_idx)


