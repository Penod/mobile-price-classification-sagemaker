# Training entry point executed inside the SageMaker SKLearn container.
# Reads CSVs from /opt/ml/input/data/{train,test}, trains a RandomForest, writes model + metrics.
import argparse, os, json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def read_csv_from_channel(channel_dir: str):
    # SageMaker mounts channel data at /opt/ml/input/data/<channel-name>/
    files = [f for f in os.listdir(channel_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV found in {channel_dir}")
    df = pd.read_csv(os.path.join(channel_dir, files[0]))
    target = "price_range" if "price_range" in df.columns else df.columns[-1]  # assume last column when unknown
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def save_metrics(y_true, y_pred, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=300)      # number of trees
    p.add_argument("--max-depth", type=int, default=10)          # tree depth
    p.add_argument("--random-state", type=int, default=42)       # reproducibility
    return p.parse_args()

def main():
    args = parse_args()
    train_dir = "/opt/ml/input/data/train"
    test_dir  = "/opt/ml/input/data/test"
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    out_dir   = os.path.join(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"), "metrics")

    X_train, y_train = read_csv_from_channel(train_dir)
    X_test,  y_test  = read_csv_from_channel(test_dir)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    save_metrics(y_test, y_pred, out_dir)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "model.joblib"))    # SageMaker will capture this as model.tar.gz

if __name__ == "__main__":
    main()
