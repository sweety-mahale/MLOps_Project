import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# Initialize DagsHub and set up MLflow experiment tracking

dagshub.init(repo_owner='sweety-mahale', repo_name='MLOps_Project', mlflow=True)
mlflow.set_experiment("Experiment 3")  # Name of the experiment in MLflow
mlflow.set_tracking_uri("https://dagshub.com/sweety-mahale/MLOps_Project.mlflow")

# Load the dataset from a CSV file
data = pd.read_csv(r"C:\Users\Sweety\OneDrive\DataScience\Loan_amount_prediction\raw_data\data_reg.csv")

# Split the dataset into training and test sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

# Define a function to fill missing values in the dataset with the median value of each column
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():  # Check if there are missing values in the column
            median_value = df[column].median()  # Calculate the median
            df[column].fillna(median_value, inplace=True)  # Fill missing values with the median
    return df

def transformation(df, is_train):
    try:
        # Exclude target column before transformation
        target_col = 'Loan Sanction Amount (USD)'  
        if target_col in df.columns:
            target = df[target_col]
            df = df.drop(columns=[target_col])
        else:
            target = None

        # Apply categorical transformations
        df['Income Stability'] = df['Income Stability'].map({'Low': 0, 'High': 1})
        df['Location'] = df['Location'].map({'Rural': 0, 'Semi-Urban': 1, 'Urban': 2})
        df['Has Active Credit Card'] = df['Has Active Credit Card'].map({'Unpossessed': 0, 'Inactive': 1, 'Active': 2})
        df['Property Location'] = df['Property Location'].map({'Rural': 0, 'Semi-Urban': 1, 'Urban': 2})

        categorical_cols = ['Gender', 'Profession', 'Type of Employment']

        if is_train:
            trnf_1 = ColumnTransformer(
                [('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)],
                remainder='passthrough'
            )
            transformed_array = trnf_1.fit_transform(df)

            with open("ohe_transformer.pkl", "wb") as f:
                pickle.dump(trnf_1, f)
        else:
            with open("ohe_transformer.pkl", "rb") as f:
                trnf_1 = pickle.load(f)
            transformed_array = trnf_1.transform(df)

        df_transformed = pd.DataFrame(transformed_array, columns=trnf_1.get_feature_names_out(), index=df.index)

        # Reattach the target column if present
        if target is not None:
            df_transformed[target_col] = target

        return df_transformed
    
    except Exception as e:
        raise Exception(f"Error transformation:{e}")


train_processed_data = transformation(train_data, True)
test_processed_data = transformation(test_data, False)


# Separate features (X) and target (y) for training
y_train = train_processed_data['Loan Sanction Amount (USD)']
X_train = train_processed_data.drop(columns=['remainder__Loan sanctioned','Loan Sanction Amount (USD)'])  # Features
  # Target variable

n_estimators = 100  # Number of trees in the Random Forest

# Start a new MLflow run for tracking the experiment
with mlflow.start_run():

    # Initialize and train the Random Forest model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Save the trained model to a file using pickle
    pickle.dump(lr, open("Linear_model.pkl", "wb"))

    # Prepare test data for prediction (features and target)
    X_test = test_processed_data.drop(columns=['remainder__Loan sanctioned','Loan Sanction Amount (USD)'])  # Features for testing
    y_test = test_processed_data['Loan Sanction Amount (USD)']  # Target variable for testing

    # Import necessary metrics for evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Load the saved model for prediction
    model = pickle.load(open('Linear_model.pkl', "rb"))

    # Predict the target for the test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)  # Accuracy
    mae = mean_absolute_error(y_test, y_pred)  # Precision
  

    # Log metrics to MLflow for tracking
    mlflow.log_metric("R2_Score", r2)
    mlflow.log_metric("mean_absolute_error", mae)
   


   



    # Log the trained model to MLflow
    mlflow.sklearn.log_model(lr, "LinearRegression")

    # Log the source code file for reference
    mlflow.log_artifact(__file__)

    # Set tags in MLflow to store additional metadata
    mlflow.set_tag("author", "Sweety")
    mlflow.set_tag("model", "LR")

    # Print out the performance metrics for reference
    print("R2_Score", r2)
    print("mean_absolute_error", mae)
