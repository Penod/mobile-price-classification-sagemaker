# Deploys a trained SKLearn model created by the Estimator.
# Assumes you have run training and have 'sklearn_estimator' saved via job name or model data S3 URI.
import argparse, boto3, sagemaker, os
from sagemaker.sklearn import SKLearnModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-data", required=True, help="s3://.../model.tar.gz produced by training")
    ap.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--endpoint", required=True, help="Endpoint name to create/update")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"))
    args = ap.parse_args()

    sess = sagemaker.Session(boto_session=boto3.Session(region_name=args.region))
    model = SKLearnModel(
        model_data=args.model_data,
        role=args.role_arn,
        framework_version="1.2-1",
        sagemaker_session=sess
    )
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=args.instance_type,
        endpoint_name=args.endpoint
    )
    print("Deployed endpoint:", predictor.endpoint_name)

if __name__ == "__main__":
    main()
