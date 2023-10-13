import numpy as np 

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = None 
        
    def fit(self, X=None, y=None):
        # Fix the seed for repeatibility
        np.random.seed(42)
        # initialize weights and bias 
        self.weights, self.bias = np.random.random(X.shape[1]), np.random.random(1)
        for _ in range(self.n_iters):
            self._update_weights(X, y)
            
        return self 
    
    def predict(self, X=None):
        return  np.dot(X, self.weights) + self.bias 
    
    def _update_weights(self, X=None, y=None):
        y_pred = np.dot(X, self.weights) + self.bias 
        n_obs = X.shape[0]
        dw = (1.0/n_obs) * np.dot(X.T, (y_pred-y))
        db = (1.0/n_obs) * np.sum(y_pred-y)
        self.weights -= self.lr * dw 
        self.bias -= self.lr*db 
        return self 
    

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate 
        self.n_iters = n_iters 
        self.weights = None 
        self.bias = None 
    
    def fit(self, X=None, y=None):
        # Fix the seed for repeatibility
        np.random.seed(42)
        # initialize weights and bias 
        self.weights = np.random.random(X.shape[1])
        self.bias = np.random.random(1) 
        
        for _ in range(self.n_iters):
            self._update_weights(X, y)
            
        return self 
    
    def predict_proba(self, X=None):
        z = np.dot(X, self.weights) + self.bias 
        return self._sigmoid(z)
    
    def predict(self, X=None, threshold=0.5):
        y_pred_proba = self.predict_proba(X) 
        return (y_pred_proba > threshold).astype(int)
    
    def _sigmoid(self, z):
        return 1.0 / (1+np.exp(-z))
    
    def _update_weights(self, X=None, y=None):
        z = np.dot(X, self.weights) + self.bias 
        y_pred = self._sigmoid(z) 
        n_obs = X.shape[0] 
        
        dw = (1.0 / n_obs) * np.dot(X.T, (y_pred-y))
        db = (1.0 / n_obs) * np.sum(y_pred-y)
        
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return self 

    