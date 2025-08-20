# Training entry point for SageMaker SKLearn container
# - Reads CSV from /opt/ml/input/data/{train,test}
# - Trains a RandomForest
# - Saves model to /opt/ml/model/model.joblib (picked up by SageMaker)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import os
import pandas as pd
import numpy as np

def model_fn(model_dir):
    """
    Load the model from the model_dir.
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the user are passed as command line arguments
    parser.add_argument("--n_estimators", type=int, default=100)  
    parser.add_argument("--random_state", type=int, default=42)

    # Data, model, and output directories
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")) 
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train_file", type=str, default="trainv1.csv")
    parser.add_argument("--test_file", type=str, default="testv1.csv")

    args, _ = parser.parse_known_args()

    print("SKLearn Version:", sklearn.__version__)
    print("Joblib Version:", joblib.__version__)

    print("[INFO] Reading training data")
    print()
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop(-1)  # Remove the label column from features

    print("Building training and testing datasets")
    print()
    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]
    y_test = test_df[label]

    print('Column order: ')
    print(features)
    print()

    print("Label column is: ", label)
    print()

    print("Data shape: ")
    print()
    print("---- SHAPE OF TRAINING DATA (85%) ----")
    print(X_train.shape)
    print(y_train.shape)
    print()


    print("Training RandomForest Model.....")
    print()
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state)
    model.fit(X_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at: " + model_path)
    print()



    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    test_confusion_matrix = confusion_matrix(y_test, y_pred_test)


    print()
    print("---- METRICS RESULTS FOR TESTING DATA (15%) ----")
    print()
    print("Total Rows are: ", X_test.shape[0])
    print("[TESTING] Model Accuracy is: ", test_accuracy)
    print("[TESTING] Classification Report: ", test_report)




