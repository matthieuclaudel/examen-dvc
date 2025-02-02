import pandas as pd
from sklearn.model_selection import train_test_split
import os

input_data = "./data/raw_data/raw.csv"
output_path = "./data/processed_data/split"

if not os.path.exists(output_path):
    os.makedirs(output_path)


def split_data(input_data, output_path):
    # Import dataset and separate features and target variable
    df_raw = pd.read_csv(input_data)
    X = df_raw.drop(columns=['silica_concentrate'], axis=1)
    y = df_raw['silica_concentrate']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    # save splitted data
    save_data_split(X_train, X_test, y_train, y_test, output_path)


def save_data_split(X_train, X_test, y_train, y_test, output_path):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = os.path.join(output_path, f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    split_data(input_data, output_path)