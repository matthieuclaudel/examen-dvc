import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def normalize_data():
    df_X_train = pd.read_csv('./data/processed_data/split/X_train.csv')
    df_X_test = pd.read_csv('./data/processed_data/split/X_test.csv')
    
    # Suppression colonne 'date'
    df_X_train.drop(columns=['date'], inplace=True)
    df_X_test.drop(columns=['date'], inplace=True)
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_X_train)
    X_test_scaled = scaler.transform(df_X_test)

    # Save normalized data
    df_X_train_scaled = pd.DataFrame(X_train_scaled, columns=df_X_train.columns)
    df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=df_X_test.columns)


    output_dir = './data/processed_data/normalize'
    os.makedirs(output_dir, exist_ok=True)

    df_X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    df_X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)

if __name__ == '__main__':
    normalize_data()