# Deletes a SageMaker endpoint + config + model to stop charges.
import argparse, boto3, sagemaker, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True, help="Endpoint name")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
    args = ap.parse_args()

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    sm = sagemaker_session.sagemaker_client

    # Delete endpoint
    try:
        sm.delete_endpoint(EndpointName=args.endpoint)
        print("Deleted endpoint:", args.endpoint)
    except sm.exceptions.ClientError as e:
        print("Endpoint delete issue:", e)

    # Delete endpoint config
    try:
        sm.delete_endpoint_config(EndpointConfigName=args.endpoint)
        print("Deleted endpoint-config:", args.endpoint)
    except sm.exceptions.ClientError as e:
        print("Endpoint-config delete issue:", e)

    # Delete model (same name convention)
    try:
        sm.delete_model(ModelName=args.endpoint)
        print("Deleted model:", args.endpoint)
    except sm.exceptions.ClientError as e:
        print("Model delete issue:", e)

if __name__ == "__main__":
    main()
