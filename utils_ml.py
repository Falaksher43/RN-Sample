import joblib
import scipy
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn import linear_model

from sklearn.base import BaseEstimator, TransformerMixin

class svdpca(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, n_components=5, n_iter=20, random_state=42):
        self._n_components = n_components
        self._n_iter = n_iter
        self._random_state = random_state

    def fit(self, X):
        self._n_components = X.shape[1]
        cur_pca = PCA(n_components=self._n_components)
        cur_pca.fit(X)
        self._pca = cur_pca
        return self

    def transform(self, X):
        return self._pca.transform(X)

