import os
from urllib.parse import urlparse

import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.processing import Processor, ProcessingInput, ProcessingOutput


class SageMakerClient:
    def __init__(
        self, aws_profile: str, aws_region: str, aws_role: str | None = None
    ) -> None:
        profile_name: str | None = aws_profile if aws_profile else None

        if profile_name:
            print("Using profile {}.".format(profile_name))
        else:
            print("Using AWS credentials from environment or default chain.")

        self.boto_session = boto3.Session(
            profile_name=profile_name, region_name=aws_region
        )

        if aws_role:
            sts_client = self.boto_session.client("sts")
            sts_client.assume_role(
                RoleArn=aws_role, RoleSessionName="EasySMSession"
            )

        self.sagemaker_session = Session(boto_session=self.boto_session)
        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.role = (
            get_execution_role(self.sagemaker_session)
            if aws_role is None
            else aws_role
        )
        self.sagemaker_client = self.boto_session.client(
            "sagemaker", region_name=aws_region
        )

    def upload_data(self, input_dir: str, s3_dir: str) -> str:
        """Uploads data to S3."""
        bucket = SageMakerClient._get_s3_bucket(s3_dir)
        prefix = SageMakerClient._get_s3_key_prefix(s3_dir) or "data"
        self.sagemaker_session.upload_data(
            path=input_dir, bucket=bucket, key_prefix=prefix
        )
        return os.path.join("s3://", bucket, prefix)

    def train(
        self,
        image_name: str,
        input_s3_data_location: str,
        train_instance_type: str,
        instance_count: int,
        output_path: str,
        base_job_name: str,
    ) -> str:
        """Train model on SageMaker."""
        image = self._construct_image_location(image_name)
        self.sagemaker_client.create_training_job(
            TrainingJobName=base_job_name,
            RoleArn=self.role,
            AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
            InputDataConfig=[
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": input_s3_data_location,
                        }
                    },
                }
            ],
            OutputDataConfig={"S3OutputPath": output_path},
            ResourceConfig={
                "InstanceType": train_instance_type,
                "InstanceCount": instance_count,
            },
        )
        return f"{output_path}/{base_job_name}/output/model.tar.gz"

    def deploy_serverless(
        self,
        image_name: str,
        s3_model_location: str,
        memory_size_in_mb: int,
        endpoint_name: str | None = None,
        max_concurrency: int = 5,
    ) -> str:
        """Deploy model to SageMaker as serverless endpoint."""
        if endpoint_name is None:
            raise ValueError("endpoint_name is required for deploy_serverless")

        model_name = self._create_model(image_name, s3_model_location)
        endpoint_config_name = f"{endpoint_name}-config"
        self._create_serverless_epc(
            endpoint_config_name, model_name, memory_size_in_mb, max_concurrency
        )
        return self._create_or_update_endpoint(endpoint_name, endpoint_config_name, "serverless")

    def deploy(
        self,
        image_name: str,
        s3_model_location: str,
        instance_type: str,
        endpoint_name: str | None = None,
        instance_count: int = 1,
    ) -> str:
        """Deploy model to SageMaker as provisioned endpoint."""
        if endpoint_name is None:
            raise ValueError("endpoint_name is required for deploy")

        model_name = self._create_model(image_name, s3_model_location)
        endpoint_config_name = f"{endpoint_name}-config"
        self._create_endpoint_config(
            endpoint_config_name, model_name, instance_type, instance_count
        )
        return self._create_or_update_endpoint(endpoint_name, endpoint_config_name, "provisioned")

    def _create_model(self, image_name: str, s3_model_location: str) -> str:
        """Create SageMaker model and return model name."""
        image = self._construct_image_location(image_name)
        model_name = f"model-{hash(s3_model_location)}"
        self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": image, "ModelDataUrl": s3_model_location},
            ExecutionRoleArn=self.role,
        )
        return model_name

    def _create_or_update_endpoint(
        self, endpoint_name: str, endpoint_config_name: str, endpoint_type: str
    ) -> str:
        """Create new endpoint or update existing one."""
        if self._check_endpoint_exists(endpoint_name):
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
            )
            print(f"Update in progress for endpoint: {endpoint_name}")
        else:
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
            )
            print(f"Creation in progress for endpoint: {endpoint_name}")
        return endpoint_name

    def _create_endpoint_config(
        self,
        endpoint_config_name: str,
        model_name: str,
        instance_type: str,
        instance_count: int,
    ) -> None:
        """Create endpoint config for provisioned inference."""
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                    "InstanceType": instance_type,
                    "InitialInstanceCount": instance_count,
                }
            ],
        )

    def _check_endpoint_exists(self, endpoint_name: str) -> bool:
        """Check if an endpoint already exists."""
        response = self.sagemaker_client.list_endpoints()
        return endpoint_name in [e["EndpointName"] for e in response.get("Endpoints", [])]

    def _create_serverless_epc(
        self,
        endpoint_config_name: str,
        model_name: str,
        memory_size_in_mb: int,
        max_concurrency: int,
    ) -> None:
        """Create endpoint config for serverless inference."""
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                    "ServerlessConfig": {
                        "MemorySizeInMB": memory_size_in_mb,
                        "MaxConcurrency": max_concurrency,
                    },
                }
            ],
        )

    def batch_transform(
        self,
        image_name: str,
        s3_model_location: str,
        s3_input_location: str,
        s3_output_location: str,
        transform_instance_count: int,
        transform_instance_type: str,
        wait: bool = False,
        job_name: str | None = None,
    ) -> str | None:
        """Execute batch transform on a trained model to SageMaker."""
        model_name = self._create_model(image_name, s3_model_location)
        job_name = job_name or "transform-job"

        self.sagemaker_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            TransformInput={
                "DataSource": {"S3DataSource": {"S3Uri": s3_input_location}},
                "ContentType": "text/csv",
                "SplitType": "Line",
            },
            TransformOutput={
                "S3OutputPath": s3_output_location,
                "Accept": "text/csv",
            },
            TransformResources={
                "InstanceType": transform_instance_type,
                "InstanceCount": transform_instance_count,
            },
        )

        if wait:
            job_description = self.sagemaker_client.describe_transform_job(
                TransformJobName=job_name
            )
            return job_description["TransformJobStatus"]
        return None

    def shutdown_endpoint(self, endpoint_name: str) -> None:
        """Shuts down a SageMaker endpoint."""
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    @staticmethod
    def _get_s3_bucket(s3_dir: str) -> str:
        """Extract bucket from S3 dir."""
        return urlparse(s3_dir).netloc

    @staticmethod
    def _get_s3_key_prefix(s3_dir: str) -> str:
        """Extract key prefix from S3 dir."""
        return urlparse(s3_dir).path.lstrip("/").rstrip("/")

    def _construct_image_location(self, image_name: str) -> str:
        """Construct full ECR image URI."""
        account = self.boto_session.client("sts").get_caller_identity()["Account"]
        region = self.boto_session.region_name
        return f"{account}.dkr.ecr.{region}.amazonaws.com/{image_name}"

    def process(
        self,
        image_name: str,
        processing_instance_type: str,
        instance_count: int,
        file: str,
        s3_input_location: str | None,
        input_sharded: bool,
        s3_output_location: str | None,
        base_job_name: str,
    ) -> None:
        """Process python file on SageMaker."""
        self._run_processing(
            image_name,
            processing_instance_type,
            instance_count,
            ["process", file],
            s3_input_location,
            input_sharded,
            s3_output_location,
            base_job_name,
        )

    def make(
        self,
        image_name: str,
        processing_instance_type: str,
        instance_count: int,
        target: str,
        s3_input_location: str | None,
        input_sharded: bool,
        s3_output_location: str | None,
        base_job_name: str,
    ) -> None:
        """Build make targets in easy_sm_base/processing on SageMaker."""
        self._run_processing(
            image_name,
            processing_instance_type,
            instance_count,
            ["make", target],
            s3_input_location,
            input_sharded,
            s3_output_location,
            base_job_name,
        )

    def _run_processing(
        self,
        image_name: str,
        processing_instance_type: str,
        instance_count: int,
        arguments: list[str],
        s3_input_location: str | None,
        input_sharded: bool,
        s3_output_location: str | None,
        base_job_name: str,
    ) -> None:
        """Run processing job with given arguments."""
        image = self._construct_image_location(image_name)
        proc = Processor(
            image_uri=image,
            role=self.role,
            instance_count=instance_count,
            instance_type=processing_instance_type,
            base_job_name=base_job_name,
            sagemaker_session=self.sagemaker_session,
        )

        proc_in = None
        if s3_input_location:
            dist_type = "ShardedByS3Key" if input_sharded else "FullyReplicated"
            proc_in = [
                ProcessingInput(
                    input_name="proc_in",
                    source=s3_input_location,
                    destination="/opt/ml/processing/input/",
                    s3_data_distribution_type=dist_type,
                )
            ]

        proc_out = None
        if s3_output_location:
            proc_out = [
                ProcessingOutput(
                    output_name="proc_out",
                    source="/opt/ml/processing/output/",
                    destination=s3_output_location,
                )
            ]

        proc.run(wait=True, arguments=arguments, inputs=proc_in, outputs=proc_out)
