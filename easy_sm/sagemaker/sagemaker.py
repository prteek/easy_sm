import os
import sagemaker as sage
from sagemaker import image_uris, payloads, model_uris
from sagemaker.processing import ProcessingInput, ProcessingOutput
from urllib.parse import urlparse
from datetime import datetime
import boto3


class SageMakerClient(object):
    def __init__(
            self,
            aws_profile,
            aws_region,
            aws_role=None
    ):
        print("Using profile {}.".format(aws_profile))
        if aws_role:
            sts_client = boto3.client('sts')
            _ = sts_client.assume_role(
                    RoleArn=aws_role,
                    RoleSessionName="EasySMSession"
                )

        self.boto_session = boto3.Session(profile_name=aws_profile, region_name=aws_region)
        self.sagemaker_session = sage.Session(boto_session=self.boto_session)

        self.aws_region = aws_region
        self.aws_profile = aws_profile
        self.role = sage.get_execution_role(self.sagemaker_session) if aws_role is None else aws_role
        self.sagemaker_client = self.boto_session.client('sagemaker', region_name=aws_region)

    def upload_data(self, input_dir, s3_dir):
        """
        Uploads data to S3
        :param input_dir: [str], local input directory where files are located
        :param s3_dir: [str], S3 directory to upload files
        :return: [str], S3 path where data are uploaded
        """
        bucket = SageMakerClient._get_s3_bucket(s3_dir)
        prefix = SageMakerClient._get_s3_key_prefix(s3_dir) or 'data'
        self.sagemaker_session.upload_data(path=input_dir, bucket=bucket, key_prefix=prefix)

        return os.path.join('s3://', bucket, prefix)

    def train(
            self,
            image_name,
            input_s3_data_location,
            train_instance_type,
            instance_count,
            output_path,
            base_job_name,
    ):
        """
        Train model on SageMaker
        :param image_name: [str], name of Docker image
        :param input_s3_data_location: [str], S3 location to input data
        :param train_instance_type: [str], ec2 instance type
        :param output_path: [str], S3 location for saving the training artefacts
        :param base_job_name: [str], Optional prefix for the SageMaker training job
        :return: [str], the model location in S3
        """
        image = self._construct_image_location(image_name)

        estimator = sage.estimator.Estimator(
            image_uri=image,
            role=self.role,
            instance_count=instance_count,
            instance_type=train_instance_type,
            input_mode='File',
            output_path=output_path,
            code_location=output_path,
            base_job_name=base_job_name,
            sagemaker_session=self.sagemaker_session,
        )

        estimator.fit(input_s3_data_location)

        return estimator.model_data

    def deploy_serverless(
            self,
            image_name,
            s3_model_location,
            memory_size_in_mb,
            endpoint_name=None,
            max_concurrency=5
    ):
        """
        Deploy model to SageMaker
        :param image_name: [str], name of Docker image
        :param s3_model_location: [str], model location in S3
        :param memory_size_in_mb: [str],
        :param endpoint_name: [optional[str]], Optional name for the SageMaker endpoint
        :param max_concurrency: [optional[int]], Optional maximum concurrency for SageMaker endpoint

        :return: [str], endpoint name
        """
        image = self._construct_image_location(image_name)
        model = sage.Model(
            model_data=s3_model_location,
            image_uri=image,
            role=self.role,
            sagemaker_session=self.sagemaker_session,
        )
        model.create()
        model_name = model.name

        if not self._check_endpoint_exists(endpoint_name):

            endpoint_config_name = f"{endpoint_name}-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
            _ = self._create_serverless_epc(
                endpoint_config_name,
                model_name,
                memory_size_in_mb,
                max_concurrency
            )

            _ = self.sagemaker_client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)

            print(f"Creation in progress for serverless endpoint: {endpoint_name}")
            return endpoint_name

        else:
            # ValueError raised if there is an endpoint already
            print(f"Serverless endpoint: {endpoint_name} already exists, updating...")
            endpoint_config_name = f"{endpoint_name}-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"

            _ = self._create_serverless_epc(
                endpoint_config_name,
                model_name,
                memory_size_in_mb,
                max_concurrency
            )

            _ = self.sagemaker_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name)

            print(f"Update in progress for serverless endpoint: {endpoint_name}")
            return endpoint_name

    def _check_endpoint_exists(self, endpoint_name: str) -> bool:
        """Check if an endpoint already exists"""
        response_blob = self.sagemaker_client.list_endpoints()
        endpoint_names = [e["EndpointName"] for e in response_blob["Endpoints"]]
        return endpoint_name in endpoint_names

    def _create_serverless_epc(
            self,
            endpoint_config_name: str,
            model_name: str,
            memory_size_in_mb: int,
            max_concurrency: int
    ):
        """Create a new End point config for serverless inference,
         that uses current model. This config is created for each deployment update as
        lineage tracking for endpoints"""

        create_endpoint_config_response = self.sagemaker_client.create_endpoint_config(
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

        return create_endpoint_config_response

    def batch_transform(
            self,
            image_name,
            s3_model_location,
            s3_input_location,
            s3_output_location,
            transform_instance_count,
            transform_instance_type,
            wait=False,
            job_name=None
    ):
        """
        Execute batch transform on a trained model to SageMaker
        :param image_name: [str], name of Docker image
        :param s3_model_location: [str], model location in S3
        :param s3_input_location: [str], S3 input data location
        :param s3_output_location: [str], S3 output data location
        :param transform_instance_count: [str],  number of ec2 instances
        :param transform_instance_type: [str], ec2 instance type
        :param wait: [bool, default=False], wait or not for the batch transform to finish
        :param job_name: [str, default=None], name for the SageMaker batch transform job

        :return: [str], transform job status if wait=True.
        Valid values: 'InProgress'|'Completed'|'Failed'|'Stopping'|'Stopped'
        """
        image = self._construct_image_location(image_name)

        model = sage.Model(
            model_data=s3_model_location,
            image_uri=image,
            role=self.role,
            sagemaker_session=self.sagemaker_session
        )

        content_type = "text/csv"

        transformer = model.transformer(
            instance_type=transform_instance_type,
            instance_count=transform_instance_count,
            output_path=s3_output_location,
            accept=content_type,
            strategy="MultiRecord",
            assemble_with="Line",
        )

        transformer.transform(data=s3_input_location, split_type='Line', content_type=content_type, job_name=job_name)

        if wait:
            try:
                transformer.wait()
            except Exception:
                # If there is an error, wait() throws an exception, and we're not able to return a Failed status
                pass
            finally:
                job_name = transformer.latest_transform_job.job_name
                job_description = self.sagemaker_client.describe_transform_job(TransformJobName=job_name)

            return job_description['TransformJobStatus']

    def shutdown_endpoint(self, endpoint_name):
        """
        Shuts down a SageMaker endpoint.
        :param endpoint_name: [str], name of the endpoint to be shut down
        """
        self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

    @staticmethod
    def _get_s3_bucket(s3_dir):
        """
        Extract bucket from S3 dir
        :param s3_dir: [str], input S3 directory
        :return: [str], extracted bucket name
        """
        return urlparse(s3_dir).netloc

    @staticmethod
    def _get_s3_key_prefix(s3_dir):
        """
        Extract key prefix from S3 dir
        :param s3_dir: [str], input S3 directory
        :return: [str], extracted key prefix name
        """
        return urlparse(s3_dir).path.lstrip('/').rstrip('/')

    def _construct_image_location(self, image_name):
        account = self.boto_session.client('sts').get_caller_identity()['Account']
        region = self.boto_session.region_name

        return '{account}.dkr.ecr.{region}.amazonaws.com/{image}'.format(
            account=account,
            region=region,
            image=image_name
        )

    def process(
            self,
            image_name,
            processing_instance_type,
            instance_count,
            file,
            s3_input_location,
            input_sharded,
            s3_output_location,
            base_job_name,

    ):
        """
        Process python file on SageMaker
        :param image_name: [str], name of Docker image
        :param train_instance_type: [str], ec2 instance type
        :param file: [str], python filename
        :param s3_input_location: [str], S3 input data location
        :param s3_output_location: [str], S3 output data location
        :param base_job_name: [str], Optional prefix for the SageMaker processing job
        :return: None
        """
        image = self._construct_image_location(image_name)

        proc = sage.processing.Processor(
            image_uri=image,
            role=self.role,
            instance_count=instance_count,
            instance_type=processing_instance_type,
            base_job_name=base_job_name,
            sagemaker_session=self.sagemaker_session,
        )

        dist_map = {True: 'ShardedByS3Key', False: 'FullyReplicated'}
        if s3_output_location:
            proc_out = [
            ProcessingOutput(
                output_name="proc_out",
                source="/opt/ml/processing/output/",
                destination=s3_output_location,
            )]
        else:
            proc_out = None

        if s3_input_location:
            proc_in = [
            ProcessingInput(
                input_name="proc_in",
                source=s3_input_location,
                destination="/opt/ml/processing/input/",
                s3_data_distribution_type=dist_map[input_sharded],
            )]
        else:
            proc_in = None

        proc.run(wait=True, arguments=['process', f'{file}'], inputs=proc_in, outputs=proc_out)

        return None


    def make(
            self,
            image_name,
            processing_instance_type,
            instance_count,
            target,
            s3_input_location,
            input_sharded,
            s3_output_location,
            base_job_name,
    ):
        """
        build make targets defined in a Makefile in easy_sm_base/processing on Sagemaker
        :param image_name: [str], name of Docker image
        :param train_instance_type: [str], ec2 instance type
        :param target: [str], target to build
        :param s3_input_location: [str], S3 input data location
        :param s3_output_location: [str], S3 output data location
        :param base_job_name: [str], Optional prefix for the SageMaker processing job
        :return: None
        """
        image = self._construct_image_location(image_name)

        proc = sage.processing.Processor(
            image_uri=image,
            role=self.role,
            instance_count=instance_count,
            instance_type=processing_instance_type,
            base_job_name=base_job_name,
            sagemaker_session=self.sagemaker_session,
        )

        dist_map = {True: 'ShardedByS3Key', False: 'FullyReplicated'}

        if s3_output_location:
            proc_out = [
            ProcessingOutput(
                output_name="proc_out",
                source="/opt/ml/processing/output/",
                destination=s3_output_location,
            )]
        else:
            proc_out = None

        if s3_input_location:
            proc_in = [
            ProcessingInput(
                input_name="proc_in",
                source=s3_input_location,
                destination="/opt/ml/processing/input/",
                s3_data_distribution_type=dist_map[input_sharded],
            )]
        else:
            proc_in = None

        proc.run(wait=True, arguments=['make', f'{target}'], inputs=proc_in, outputs=proc_out)

        return None