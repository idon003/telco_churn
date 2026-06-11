import numpy as np

class Node:
    def __init__(self, feature, threshold, left, right, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class LogisticRegression:
    def __init__(self, learning_rate = 0.01, n_iters = 1000): # initialize the model with learning rate and number of iterations
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def sigmoid(self, z): # for converting linear output to a value between 0 and 1, which can be interpreted as a probability
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y): # for training the model, it takes the training data and labels as input, and updates the weights and bias using gradient descent
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.w) + self.b
            y_hat = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
    def predict(self, X): # for making predictions, it takes the test data as input and returns the predicted labels
        X = np.asarray(X, dtype=np.float64)
        linear_model = np.dot(X, self.w) + self.b
        y_hat = self.sigmoid(linear_model)
        y_predicted = [1 if i > 0.5 else 0 for i in y_hat]
        return np.array(y_predicted)
    
class DecisionTree:

    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, x, y):
        self.root = self._build_tree(x, y)
        
    def predict(self):
        pass

    def _build_tree(self, x, y, depth=0):
        pass

    def entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        return entropy
        