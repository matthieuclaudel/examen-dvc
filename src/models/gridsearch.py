import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
import joblib
import os

X_train = pd.read_csv('./data/processed_data/normalize/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/split/y_train.csv').values.ravel()

# dictionnaire de modèles à tester
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet()
}

# Grille de paramètres pour chaque modèle
param_grid = [
    {
        'model': [LinearRegression()],
        'model__fit_intercept': [True, False],
        'model__copy_X': [True, False],
        'model__n_jobs': [None, -1],
        'model__positive': [True, False]
    },
    {
        'model': [Ridge()],
        'model__alpha': [0.1, 1, 10, 100],
        'model__fit_intercept': [True, False],
        'model__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
    },
    {
        'model': [Lasso()],
        'model__alpha': [0.1, 1, 10, 100],
        'model__fit_intercept': [True, False],
        'model__selection': ['cyclic', 'random']
    },
    {
        'model': [ElasticNet()],
        'model__alpha': [0.1, 1, 10, 100],
        'model__l1_ratio': [0.1, 0.5, 0.9],
        'model__fit_intercept': [True, False],
        'model__selection': ['cyclic', 'random']
    }
]

# Pipeline
pipeline = Pipeline([
    ('model', LinearRegression())
    ])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

# Print best model
print(grid_search.best_estimator_)

# Sauvegarde du meilleur modèle
os.makedirs('./models/params', exist_ok=True)
best_params = grid_search.best_params_
joblib.dump(best_params, './models/params/best_params.pkl')

if __name__ == '__main__':
    pass