import numpy as np 
from collections import Counter 

class KNNClassifier:
    def __init__(self, n_neighbors=5, distance="euclidian"):
        self.n_neighbors = n_neighbors 
        self.distance = self._get_distance(distance)

    def _get_distance(self, distance = "euclidian"):
        if distance == "euclidian":
            return self._euclidian_distance
        elif distance == "manhattan":
            return self._manhattan_distance
        else:
            raise ValueError("Please specify 'euclidian' or 'manhattan' metrics only")
        
    def fit(self, X=None, y=None):
        self.X_train = X 
        self.y_train = y  
    
    def predict(self, X=None):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        # predict for a single observation
        distances = [self.distance(x, x_train) for x_train in self.X_train]
        k_ind = np.argsort(distances)[:self.n_neighbors]
        labels = self.y_train[k_ind] 
        # vote 
        return Counter(labels).most_common()[0][0]
        
    def _euclidian_distance(self, p1, p2):
        return np.sqrt( np.sum((p1 - p2)**2) )

    def _manhattan_distance(self, p1, p2):
        return np.sum( np.abs(p1-p2) ) 