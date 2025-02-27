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
from sklearn.tree import DecisionTreeClassifier

# Initialize DagsHub and set up MLflow experiment tracking

dagshub.init(repo_owner='sweety-mahale', repo_name='MLOps_Project', mlflow=True)
mlflow.set_experiment("Experiment 1")  # Name of the experiment in MLflow
mlflow.set_tracking_uri("https://dagshub.com/sweety-mahale/MLOps_Project.mlflow")

# Load the dataset from a CSV file
data = pd.read_csv(r"C:\Users\Sweety\OneDrive\DataScience\Loan_amount_prediction\raw_data\data_clf.csv")

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

# Fill missing values in both the training and test datasets using the median
train_processed_data = transformation(train_data, True)
test_processed_data = transformation(test_data, False)

# Import RandomForestClassifier and pickle for model saving
from sklearn.ensemble import RandomForestClassifier
import pickle

# Separate features (X) and target (y) for training
X_train = train_processed_data.drop(columns=['Loan sanctioned'], axis=1)  # Features
y_train = train_processed_data['Loan sanctioned']  # Target variable

n_estimators = 100  # Number of trees in the Random Forest

# Start a new MLflow run for tracking the experiment
with mlflow.start_run():

    # Initialize and train the Random Forest model
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)

    # Save the trained model to a file using pickle
    pickle.dump(clf, open("Dec_tree_model.pkl", "wb"))

    # Prepare test data for prediction (features and target)
    X_test = test_processed_data.iloc[:, 0:-1].values  # Features for testing
    y_test = test_processed_data.iloc[:, -1].values  # Target variable for testing

    # Import necessary metrics for evaluation
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Load the saved model for prediction
    model = pickle.load(open('Dec_tree_model.pkl', "rb"))

    # Predict the target for the test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    acc = accuracy_score(y_test, y_pred)  # Accuracy
    precision = precision_score(y_test, y_pred)  # Precision
    recall = recall_score(y_test, y_pred)  # Recall
    f1 = f1_score(y_test, y_pred)  # F1-score

    # Log metrics to MLflow for tracking
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1)

    # Generate a confusion matrix to visualize model performance
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True)  # Visualize confusion matrix
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save the confusion matrix plot as a PNG file
    plt.savefig("confusion_matrix.png")

    # Log the confusion matrix image to MLflow
    mlflow.log_artifact("confusion_matrix.png")

    # Log the trained model to MLflow
    mlflow.sklearn.log_model(clf, "DecisionTreeClassifier")

    # Log the source code file for reference
    mlflow.log_artifact(__file__)

    # Set tags in MLflow to store additional metadata
    mlflow.set_tag("author", "Sweety")
    mlflow.set_tag("model", "DT")

    # Print out the performance metrics for reference
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)