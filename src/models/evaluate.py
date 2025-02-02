import pandas as pd 
import numpy as np
import joblib
import json
from pathlib import Path
import os

from sklearn.metrics import r2_score, mean_squared_error,accuracy_score

X_test = pd.read_csv('data/processed_data/normalize/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/split/y_test.csv').values.ravel()


# Prédictions
model = joblib.load('./models/model/model.pkl')
y_pred = model.predict(X_test)   
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
metrics = { "mse": mse,
           "r2": r2}

# Enregistrement des métriques
os.makedirs('./metrics', exist_ok=True)
accuracy_path = Path('./metrics/scores.json')
accuracy_path.write_text(json.dumps(metrics))

# Enregistrement des prédictions dans un dataframe
df_predictions = pd.DataFrame()
df_predictions = X_test
df_predictions['Valeur réelle'] = y_test
df_predictions['Prédiction'] = y_pred

os.makedirs('./data/evaluation', exist_ok=True)
df_predictions.to_csv('data/evaluation/predictions.csv', index=False)

if __name__ == "__main__":
    pass