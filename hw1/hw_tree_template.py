import numpy as np
from collections import Counter
import random

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
        if (len(y) < self.min_samples) or self.gini_impurity(y) == 0: #create a leaf
            return TreeNode(X, y)


        dL, yL, dR, yR, criteria, idx = self.split(X, y, self.get_candidate_columns(X, self.rand))
        node = TreeNode(X, y, criteria, idx)
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
                
                quality_of_split = len(yL) / l * l_impurity + len(yR) / l * r_impurity
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

    def __init__(self, X = None, y = None, criteria = None, idx = None, L = None, R = None):
        self.X = X
        self.y = y
        self.criteria = criteria
        self.idx = idx
        self.value = Counter(y).most_common(1)[0][0]
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
    
    def print_tree(self, depth=0):
        # print(self.criteria, self.idx, self.value)
        # print(self.L, self.R)
        if(self.criteria == None):
            # print("  " * depth + "|—>  " + str(self.y))
            return
        print("  " * depth + "|—>  t: " + str(self.criteria) + "  f_idx: " + str(self.idx))
        if self.L:
            self.L.print_tree(depth + 1)
        if self.R:
            self.R.print_tree(depth + 1)

class RandomForest:

    def __init__(self, rand=random.Random(), n=50):
        self.n = n
        self.rand = rand

    def build(self, X, y):
        trees = []
        for seed in [int(self.rand.uniform(0, 1) * 1000) for _ in range(self.n)] :
            r = random.Random(seed)
            t = Tree(rand=r, get_candidate_columns=random_sqrt_columns)

            bootstrap_indices = r.sample(range(len(y)), len(y))
            X_b = X[bootstrap_indices]
            y_b = y[bootstrap_indices]

            trees.append(t.build(X_b, y_b))

        return RFModel(trees=trees)


class RFModel:

    def __init__(self, trees= None):
        self.trees = trees

    def predict(self, X):
        y = []

        for tree in self.trees:
            y.append(tree.predict(X))

        y = np.array(y).T
        y = np.array([Counter(row).most_common(1)[0][0] for row in y])
        
        return y

    def importance(self):
        imps = np.zeros(self.X.shape[1])
        # ...
        return imps


if __name__ == "__main__":
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
