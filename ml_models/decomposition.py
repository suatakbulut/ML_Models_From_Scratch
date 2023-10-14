import numpy as np


class PCA:
    """ 
    Unsupervised learning method, reduce the dimensionality by mapping it 
    into lower dimension without losing too much information

    - transformed features are linearly independent 
    - dimensionality can be reduced by taking only the dimension with highest importance 
    - newly foudn dimensions should minimize the projection error 
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
        eigenvalues = eigenvalues[idxs]
        eigenvectos = eigenvectos[idxs] 

        # select the first n_componenets
        self.components = eigenvectos[:self.n_components]

    def transform(self, X):
        X = X-self.mean 
        return np.dot(X, self.components.T) 