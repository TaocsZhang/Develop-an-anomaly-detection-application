from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class scaling(BaseEstimator, TransformerMixin):
    # Scaling variables

    def __init__(self, variables):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X):
        # so that we do not over-write the original dataframe
        X = X.copy()

        for variable in self.variables:
            std_scaler = StandardScaler()
            scale_var = std_scaler.fit(X[variable].values.reshape(-1, 1))
            X[variable] = scale_var.transform(X[variable].values.reshape(-1, 1))
        return X