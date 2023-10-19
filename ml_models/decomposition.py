import numpy as np


class PCA:
    """ 
    Unsupervised learning method, reduce the dimensionality by mapping it 
    into lower dimension without losing too much information

    - transformed features are linearly independent 
    - dimensionality can be reduced by taking only the dimension with highest importance 
    - newly found dimensions should minimize the projection error 
    - projected point should have maximum spread 
    """

    def __init__(self, n_components):
        self.n_components = n_components 
        self.components = None 
        self.mean = None 


    def fit(self, X):
        # mean centering 
        self.mean = np.mean(X, axis=0)
        X = X - self.mean 

        # covariance, function needs sampels as columsn 
        cov = np.cov(X.T)

        # eigenvectos and eigenvalues
        eigenvectos, eigenvalues = np.linalg.eig(cov) 

        # convert it to row veector
        eigenvectos = eigenvectos.T

        # sort eigenvectors
        idxs = np.argsort(eigenvalues)[::-1] 
        eigenvectos = eigenvectos[idxs] 

        # select the first n_components of the eigenvectors 
        self.components = eigenvectos[:self.n_components]

    def transform(self, X):
        X = X-self.mean 
        return np.dot(X, self.components.T) 
    

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from metrics import mean_square_error, accuracy 

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Before PCA
    print("\nTesting PCA on Sklearn Logistic Regression")
    print("==========================================")
    from sklearn.linear_model import LogisticRegression
    sk_logit = LogisticRegression(max_iter=10000)
    sk_logit.fit(X_train, y_train) 
    y_pred = sk_logit.predict(X_test)
    print("Sklearn Accuracy without PCA (30 features):", accuracy(y_test, y_pred))

    pca = PCA(n_components=20)
    pca.fit(X_train) 
    X_train = pca.transform(X_train) 
    X_test = pca.transform(X_test)  

    sk_logit_pca = LogisticRegression(max_iter=10000)
    sk_logit_pca.fit(X_train, y_train) 
    y_pred = sk_logit_pca.predict(X_test)
    print("Sklearn Accuracy with PCA (20 components) :", accuracy(y_test, y_pred))
