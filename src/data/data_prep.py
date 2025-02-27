import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pickle

def load_data(filepath : str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")
# train_data = pd.read_csv("./data/raw/train.csv")
# test_data = pd.read_csv("./data/raw/test.csv")


def fill_missing_with_mean(df):
    try:
        for column in df.columns:
            if df[column].isnull().any():
                mean_value = df[column].mean()
                df[column].fillna(mean_value,inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error Filling missing values with mean:{e}")
    
def transformation(df, is_train):
    try:
        df['Income Stability'] = df['Income Stability'].map({'Low': 0, 'High': 1})
        df['Location'] = df['Location'].map({'Rural': 0, 'Semi-Urban': 1, 'Urban': 2})
        df['Has Active Credit Card'] = df['Has Active Credit Card'].map({'Unpossessed': 0, 'Inactive': 1, 'Active': 2})
        df['Property Location'] = df['Property Location'].map({'Rural': 0, 'Semi-Urban': 1, 'Urban': 2})

        # Define categorical columns for OneHotEncoding
        categorical_cols = ['Gender', 'Profession', 'Type of Employment']

        # Check if we are processing training data
        if is_train:
            # Fit OneHotEncoder on training data
            trnf_1 = ColumnTransformer(
                [('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)],
                remainder='passthrough'
            )
            transformed_array = trnf_1.fit_transform(df)

            # Save the trained transformer
            with open("ohe_transformer.pkl", "wb") as f:
                pickle.dump(trnf_1, f)
        else:
            # Load the saved OneHotEncoder for test data
            with open("ohe_transformer.pkl", "rb") as f:
                trnf_1 = pickle.load(f)
            transformed_array = trnf_1.transform(df)

        # Convert NumPy array back to DataFrame
        new_feature_names = trnf_1.get_feature_names_out()
        df_transformed = pd.DataFrame(transformed_array, columns=new_feature_names)

        # Ensure indices match original DataFrame
        df_transformed.index = df.index  

        # Apply Label Encoding on 'Loan sanctioned'
        if 'Loan sanctioned' in df.columns:
            le = LabelEncoder()
            if is_train:
                df_transformed['Loan sanctioned'] = le.fit_transform(df['Loan sanctioned'])
                with open("label_encoder.pkl", "wb") as f:
                    pickle.dump(le, f)
            else:
                with open("label_encoder.pkl", "rb") as f:
                    le = pickle.load(f)
                df_transformed['Loan sanctioned'] = le.transform(df['Loan sanctioned'])

        return df_transformed
    
    except Exception as e:
        raise Exception(f"Error transformation:{e}")

def save_data(df : pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}:{e}")

# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data = fill_missing_with_median(test_data)

def main():
    try:
        raw_data_path = "./data/raw/"
        processed_data_path = "./data/processed"

        train_data = load_data(os.path.join(raw_data_path,"train.csv"))
        test_data = load_data(os.path.join(raw_data_path,"test.csv"))

        # train_processed_data = fill_missing_with_mean(train_data)
        # test_processed_data = fill_missing_with_mean(test_data)

        train_processed_data = transformation(train_data, True)
        test_processed_data = transformation(test_data, False)


    # data_path= os.path.join("data","processed")

        os.makedirs(processed_data_path)

        save_data(train_processed_data,os.path.join(processed_data_path,"train_processed.csv"))
        save_data(test_processed_data,os.path.join(processed_data_path,"test_processed.csv"))
    except Exception as e:
        raise Exception(f"An error occurred :{e}")
    
if __name__ == "__main__":
    main()