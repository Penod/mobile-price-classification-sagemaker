# Small client to call a deployed SageMaker endpoint using CSV-serialized rows.
# Usage: python client_predict.py --endpoint <name> --csv-path test.csv
import argparse, boto3, os, sys
from io import BytesIO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="SageMaker endpoint name")
    ap.add_argument("--csv-path", required=True, help="Path to CSV with rows to predict (no header)")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
    args = ap.parse_args()

    smrt = boto3.client("sagemaker-runtime", region_name=args.region)
    with open(args.csv_path, "rb") as f:
        payload = f.read()  # send raw CSV bytes

    resp = smrt.invoke_endpoint(
        EndpointName=args.endpoint,
        ContentType="text/csv",
        Body=payload
    )
    print(resp["Body"].read().decode("utf-8"))

if __name__ == "__main__":
    main()
