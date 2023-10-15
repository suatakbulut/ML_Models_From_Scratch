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


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from metrics import mean_square_error, accuracy 

    #####################
    # Linear Regression
    #####################
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


    lin = LinearRegression()
    lin.fit(X_train, y_train) 
    y_pred = lin.predict(X_test)
    print("\nTesting Linear Regression")
    print("=========================")
    print(" Custom MSE:", mean_square_error(y_test, y_pred))

    from sklearn.linear_model import LinearRegression
    sk_lin = LinearRegression()
    sk_lin.fit(X_train, y_train) 
    y_pred = sk_lin.predict(X_test)
    print("Sklearn MSE:", mean_square_error(y_test, y_pred))


    #####################
    # Logistic Regression
    #####################

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    logit = LogisticRegression(learning_rate=0.003, n_iters=10000)
    logit.fit(X_train, y_train) 
    y_pred = logit.predict(X_test)
    print("\nTesting Logistic Regression")
    print("===========================")
    print(" Custom Accuracy:", accuracy(y_test, y_pred))


    from sklearn.linear_model import LogisticRegression
    sk_logit = LogisticRegression(max_iter=10000)
    sk_logit.fit(X_train, y_train) 
    y_pred = sk_logit.predict(X_test)
    print("Sklearn Accuracy:", accuracy(y_test, y_pred))