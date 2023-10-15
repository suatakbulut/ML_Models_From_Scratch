import numpy as np 


class NaiveBayes:
    """ 
    Training:
        Calculate mean, var, and priors
    Prediction:
        Calculate posterior for each class:
            log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xk|y)) + log(P(y))
        Choose class with highest probability
    """

    def fit(self, X=None, y=None):
        n_samples, n_features = X.shape 
        self._classes = np.unique(y) 
        n_classes = len(self._classes)

        # calculate mean and variance for x_i|y and the prob of y for each y
        self._means  = np.zeros((n_classes, n_features), dtype=np.float64)
        self._vars   = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64) 

        for idx, c in enumerate(self._classes):
            X_c = X[y==c] 
            self._means[idx,:] = X_c.mean(axis=0)
            self._vars[idx,:] = X_c.var(axis=0)
            self._priors[idx] = len(X_c) / float(n_samples) 

    def predict(self, X=None):
        return np.array([self._predict(x) for x in X])
    

    def _predict(self, x):
        # calcualte posterior:
        # log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xk|y)) + log(P(y)) 
        posteriors = []

        for class_idx in range(len(self._classes)):
            prior = np.sum(np.log(self._pdf(class_idx, x)))
            posterior = prior + np.log(self._priors[class_idx])
            posteriors.append(posterior) 

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        numer = np.exp(-((x-self._means[class_idx])**2)/(2 * self._vars[class_idx]))
        denom = np.sqrt(2 * np.pi * self._vars[class_idx]) 

        return numer / denom 

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from metrics import mean_square_error, accuracy 

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    nb = NaiveBayes()
    nb.fit(X_train, y_train) 
    y_pred = nb.predict(X_test)
    print("\nTesting Naive Bayes Classifier")
    print("==============================")
    print(" Custom Accuracy:", accuracy(y_test, y_pred))

    from sklearn.naive_bayes import GaussianNB
    sk_nb = GaussianNB()
    sk_nb.fit(X_train, y_train) 
    y_pred = sk_nb.predict(X_test)
    print("Sklearn Accuracy:", accuracy(y_test, y_pred))
