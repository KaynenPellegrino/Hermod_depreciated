# src/modules/cloud_integrations/cloud_service_manager.py

import os
import logging
from typing import Dict, Any, Optional
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.staging import NotificationManager

# AWS Imports
import boto3
from botocore.exceptions import ClientError

# Azure Imports
from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
from azure.core.exceptions import AzureError

# GCP Imports
from google.cloud import compute_v1
from google.cloud import storage
from google.api_core.exceptions import GoogleAPICallError

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/cloud_service_manager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class CloudServiceManager:
    """
    Cross-Cloud Service Management
    Manages cloud resources, deployments, and data storage across multiple cloud platforms (AWS, Azure, GCP).
    Acts as the central hub for API calls, managing cloud compute resources, and orchestrating deployments across different environments.
    """

    def __init__(self):
        """
        Initializes the CloudServiceManager with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_cloud_config()
            self.aws_client = None
            self.azure_client = None
            self.gcp_client = None
            self.initialize_clients()
            logger.info("CloudServiceManager initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize CloudServiceManager: {e}")
            raise e

    def load_cloud_config(self):
        """
        Loads cloud configurations from the configuration manager or environment variables.
        """
        logger.info("Loading cloud configurations.")
        try:
            self.cloud_config = {
                'aws_region': self.config_manager.get('AWS_REGION', os.environ.get('AWS_DEFAULT_REGION')),
                'azure_subscription_id': self.config_manager.get('AZURE_SUBSCRIPTION_ID', os.environ.get('AZURE_SUBSCRIPTION_ID')),
                'gcp_project_id': self.config_manager.get('GCP_PROJECT_ID', os.environ.get('GCP_PROJECT_ID')),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Cloud configurations loaded: {self.cloud_config}")
        except Exception as e:
            logger.error(f"Failed to load cloud configurations: {e}")
            raise e

    def initialize_clients(self):
        """
        Initializes clients for AWS, Azure, and GCP.
        """
        logger.info("Initializing cloud clients.")
        try:
            # Initialize AWS client
            self.aws_client = boto3.client('ec2', region_name=self.cloud_config['aws_region'])
            logger.info("AWS client initialized.")

            # Initialize Azure client
            credential = DefaultAzureCredential()
            self.azure_client = ComputeManagementClient(
                credential=credential,
                subscription_id=self.cloud_config['azure_subscription_id']
            )
            logger.info("Azure client initialized.")

            # Initialize GCP clients
            self.gcp_compute_client = compute_v1.InstancesClient()
            self.gcp_storage_client = storage.Client()
            logger.info("GCP clients initialized.")

        except Exception as e:
            logger.error(f"Failed to initialize cloud clients: {e}")
            raise e

    # --------------------- AWS Methods --------------------- #

    def create_aws_instance(self, instance_params: Dict[str, Any]) -> Optional[str]:
        """
        Creates an AWS EC2 instance with the specified parameters.

        :param instance_params: Dictionary of instance parameters.
        :return: Instance ID if successful, None otherwise.
        """
        logger.info("Creating AWS EC2 instance.")
        try:
            response = self.aws_client.run_instances(**instance_params)
            instance_id = response['Instances'][0]['InstanceId']
            logger.info(f"AWS EC2 instance created with ID: {instance_id}")
            return instance_id
        except ClientError as e:
            logger.error(f"Failed to create AWS EC2 instance: {e}")
            return None

    def terminate_aws_instance(self, instance_id: str) -> bool:
        """
        Terminates the specified AWS EC2 instance.

        :param instance_id: The ID of the instance to terminate.
        :return: True if successful, False otherwise.
        """
        logger.info(f"Terminating AWS EC2 instance with ID: {instance_id}")
        try:
            self.aws_client.terminate_instances(InstanceIds=[instance_id])
            logger.info(f"AWS EC2 instance {instance_id} terminated.")
            return True
        except ClientError as e:
            logger.error(f"Failed to terminate AWS EC2 instance: {e}")
            return False

    def create_aws_s3_bucket(self, bucket_name: str) -> bool:
        """
        Creates an AWS S3 bucket.

        :param bucket_name: The name of the bucket to create.
        :return: True if successful, False otherwise.
        """
        logger.info(f"Creating AWS S3 bucket: {bucket_name}")
        s3_client = boto3.client('s3')
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"AWS S3 bucket '{bucket_name}' created.")
            return True
        except ClientError as e:
            logger.error(f"Failed to create AWS S3 bucket: {e}")
            return False

    # --------------------- Azure Methods --------------------- #

    def create_azure_vm(self, vm_params: Dict[str, Any]) -> Optional[str]:
        """
        Creates an Azure Virtual Machine with the specified parameters.

        :param vm_params: Dictionary of VM parameters.
        :return: VM name if successful, None otherwise.
        """
        logger.info("Creating Azure Virtual Machine.")
        try:
            async_vm_creation = self.azure_client.virtual_machines.begin_create_or_update(
                resource_group_name=vm_params['resource_group'],
                vm_name=vm_params['vm_name'],
                parameters=vm_params['parameters']
            )
            vm_result = async_vm_creation.result()
            logger.info(f"Azure VM '{vm_params['vm_name']}' created.")
            return vm_params['vm_name']
        except AzureError as e:
            logger.error(f"Failed to create Azure VM: {e}")
            return None

    def delete_azure_vm(self, resource_group: str, vm_name: str) -> bool:
        """
        Deletes the specified Azure Virtual Machine.

        :param resource_group: The resource group of the VM.
        :param vm_name: The name of the VM to delete.
        :return: True if successful, False otherwise.
        """
        logger.info(f"Deleting Azure VM '{vm_name}' in resource group '{resource_group}'.")
        try:
            async_vm_delete = self.azure_client.virtual_machines.begin_delete(
                resource_group_name=resource_group,
                vm_name=vm_name
            )
            async_vm_delete.result()
            logger.info(f"Azure VM '{vm_name}' deleted.")
            return True
        except AzureError as e:
            logger.error(f"Failed to delete Azure VM: {e}")
            return False

    # --------------------- GCP Methods --------------------- #

    def create_gcp_instance(self, instance_params: Dict[str, Any]) -> Optional[str]:
        """
        Creates a GCP Compute Engine instance with the specified parameters.

        :param instance_params: Dictionary of instance parameters.
        :return: Instance name if successful, None otherwise.
        """
        logger.info("Creating GCP Compute Engine instance.")
        try:
            operation = self.gcp_compute_client.insert(
                project=self.cloud_config['gcp_project_id'],
                zone=instance_params['zone'],
                instance_resource=instance_params['instance_resource']
            )
            operation.result()
            instance_name = instance_params['instance_resource']['name']
            logger.info(f"GCP instance '{instance_name}' created.")
            return instance_name
        except GoogleAPICallError as e:
            logger.error(f"Failed to create GCP instance: {e}")
            return None

    def delete_gcp_instance(self, zone: str, instance_name: str) -> bool:
        """
        Deletes the specified GCP Compute Engine instance.

        :param zone: The zone of the instance.
        :param instance_name: The name of the instance to delete.
        :return: True if successful, False otherwise.
        """
        logger.info(f"Deleting GCP instance '{instance_name}' in zone '{zone}'.")
        try:
            operation = self.gcp_compute_client.delete(
                project=self.cloud_config['gcp_project_id'],
                zone=zone,
                instance=instance_name
            )
            operation.result()
            logger.info(f"GCP instance '{instance_name}' deleted.")
            return True
        except GoogleAPICallError as e:
            logger.error(f"Failed to delete GCP instance: {e}")
            return False

    def create_gcp_storage_bucket(self, bucket_name: str) -> bool:
        """
        Creates a GCP Cloud Storage bucket.

        :param bucket_name: The name of the bucket to create.
        :return: True if successful, False otherwise.
        """
        logger.info(f"Creating GCP Cloud Storage bucket: {bucket_name}")
        try:
            bucket = self.gcp_storage_client.bucket(bucket_name)
            bucket.location = 'US'
            self.gcp_storage_client.create_bucket(bucket)
            logger.info(f"GCP bucket '{bucket_name}' created.")
            return True
        except GoogleAPICallError as e:
            logger.error(f"Failed to create GCP bucket: {e}")
            return False

    # --------------------- Orchestration Methods --------------------- #

    def deploy_application(self, cloud_provider: str, deployment_params: Dict[str, Any]) -> bool:
        """
        Orchestrates the deployment of an application to the specified cloud provider.

        :param cloud_provider: The cloud provider ('aws', 'azure', 'gcp').
        :param deployment_params: Dictionary of deployment parameters.
        :return: True if successful, False otherwise.
        """
        logger.info(f"Deploying application to {cloud_provider.upper()}.")
        try:
            if cloud_provider == 'aws':
                # Implement AWS deployment logic
                instance_id = self.create_aws_instance(deployment_params['instance_params'])
                if instance_id:
                    logger.info(f"Application deployed to AWS EC2 instance {instance_id}.")
                    return True
            elif cloud_provider == 'azure':
                # Implement Azure deployment logic
                vm_name = self.create_azure_vm(deployment_params['vm_params'])
                if vm_name:
                    logger.info(f"Application deployed to Azure VM '{vm_name}'.")
                    return True
            elif cloud_provider == 'gcp':
                # Implement GCP deployment logic
                instance_name = self.create_gcp_instance(deployment_params['instance_params'])
                if instance_name:
                    logger.info(f"Application deployed to GCP instance '{instance_name}'.")
                    return True
            else:
                logger.error(f"Unsupported cloud provider: {cloud_provider}")
                return False
        except Exception as e:
            logger.error(f"Failed to deploy application to {cloud_provider.upper()}: {e}")
            return False

    # --------------------- Notification Method --------------------- #

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.cloud_config['notification_recipients']
            if recipients:
                self.notification_manager.send_notification(
                    recipients=recipients,
                    subject=subject,
                    message=message
                )
                logger.info("Notification sent successfully.")
            else:
                logger.warning("No notification recipients configured.")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")


# --------------------- Example Usage --------------------- #

def example_usage():
    """
    Demonstrates example usage of the CloudServiceManager class.
    """
    try:
        # Initialize CloudServiceManager
        manager = CloudServiceManager()

        # AWS example
        aws_instance_params = {
            'ImageId': 'ami-0abcdef1234567890',  # Replace with a valid Image ID
            'InstanceType': 't2.micro',
            'MinCount': 1,
            'MaxCount': 1,
        }
        aws_instance_id = manager.create_aws_instance(aws_instance_params)
        if aws_instance_id:
            # Terminate the instance
            manager.terminate_aws_instance(aws_instance_id)

        # Azure example
        azure_vm_params = {
            'resource_group': 'myResourceGroup',
            'vm_name': 'myVM',
            'parameters': {
                # Define VM parameters here
            }
        }
        azure_vm_name = manager.create_azure_vm(azure_vm_params)
        if azure_vm_name:
            # Delete the VM
            manager.delete_azure_vm(azure_vm_params['resource_group'], azure_vm_name)

        # GCP example
        gcp_instance_params = {
            'zone': 'us-central1-a',
            'instance_resource': {
                'name': 'my-instance',
                # Define instance resource parameters here
            }
        }
        gcp_instance_name = manager.create_gcp_instance(gcp_instance_params)
        if gcp_instance_name:
            # Delete the instance
            manager.delete_gcp_instance(gcp_instance_params['zone'], gcp_instance_name)

        # Deploy application example
        deployment_params = {
            'instance_params': aws_instance_params  # Or azure_vm_params, gcp_instance_params
        }
        manager.deploy_application('aws', deployment_params)

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")


# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the cloud service manager example
    example_usage()
