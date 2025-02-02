import pandas as pd
from sklearn.linear_model import Ridge
import joblib
import os

X_train = pd.read_csv('./data/processed_data/normalize/X_train_scaled.csv')
y_train = pd.read_csv('./data/processed_data/split/y_train.csv').values.ravel()

# chargement des paramètres optimaux
best_params = joblib.load('./models/params/best_params.pkl')

# Filtrer les paramètres pour ne conserver que ceux pertinents pour Ridge
ridge_params = {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}


# Instanciation
model = Ridge(**ridge_params)

# Entrainement
model.fit(X_train, y_train)

# Sauvegarde du modèle
os.makedirs('./models/model', exist_ok=True)
joblib.dump(model, './models/model/model.pkl')
