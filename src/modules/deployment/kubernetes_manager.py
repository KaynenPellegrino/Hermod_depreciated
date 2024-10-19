# src/modules/deployment/kubernetes_manager.py

import os
import logging
from typing import Optional, Dict, Any
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/kubernetes_manager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class KubernetesManager:
    """
    Kubernetes Deployment Management
    Automates the deployment of applications to Kubernetes clusters, managing pods, services, deployments, and scaling.
    Interacts with Kubernetes APIs to orchestrate containerized applications.
    """

    def __init__(self):
        """
        Initializes the KubernetesManager with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_kubernetes_config()
            self.load_kube_config()
            logger.info("KubernetesManager initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize KubernetesManager: {e}")
            raise e

    def load_kubernetes_config(self):
        """
        Loads Kubernetes configurations from the configuration manager or environment variables.
        """
        logger.info("Loading Kubernetes configurations.")
        try:
            self.kubernetes_config = {
                'namespace': self.config_manager.get('KUBERNETES_NAMESPACE', 'default'),
                'deployment_name': self.config_manager.get('KUBERNETES_DEPLOYMENT_NAME', 'hermod-deployment'),
                'service_name': self.config_manager.get('KUBERNETES_SERVICE_NAME', 'hermod-service'),
                'container_image': self.config_manager.get('KUBERNETES_CONTAINER_IMAGE', 'hermod_app:latest'),
                'replicas': int(self.config_manager.get('KUBERNETES_REPLICAS', 1)),
                'container_port': int(self.config_manager.get('KUBERNETES_CONTAINER_PORT', 80)),
                'environment_variables': self.config_manager.get('ENVIRONMENT_VARIABLES', {}),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
                'kube_config_path': self.config_manager.get('KUBE_CONFIG_PATH', '~/.kube/config'),
            }
            logger.info(f"Kubernetes configurations loaded: {self.kubernetes_config}")
        except Exception as e:
            logger.error(f"Failed to load Kubernetes configurations: {e}")
            raise e

    def load_kube_config(self):
        """
        Loads Kubernetes configuration from kubeconfig file or in-cluster configuration.
        """
        logger.info("Loading Kubernetes kubeconfig.")
        try:
            kube_config_path = os.path.expanduser(self.kubernetes_config.get('kube_config_path', ''))
            if os.path.exists(kube_config_path):
                config.load_kube_config(config_file=kube_config_path)
                logger.info(f"Kubeconfig loaded from '{kube_config_path}'.")
            else:
                config.load_incluster_config()
                logger.info("Kubeconfig loaded from in-cluster configuration.")
            self.apps_v1_api = client.AppsV1Api()
            self.core_v1_api = client.CoreV1Api()
        except Exception as e:
            logger.error(f"Failed to load Kubernetes configuration: {e}")
            raise e

    def create_deployment(self):
        """
        Creates a Kubernetes Deployment.
        """
        logger.info("Creating Kubernetes Deployment.")
        try:
            deployment = self._build_deployment_object()
            response = self.apps_v1_api.create_namespaced_deployment(
                namespace=self.kubernetes_config['namespace'],
                body=deployment
            )
            logger.info(f"Deployment created. Status: {response.metadata.name}")
        except ApiException as e:
            if e.status == 409:
                logger.warning("Deployment already exists. Updating deployment.")
                self.update_deployment()
            else:
                logger.error(f"Failed to create deployment: {e}")
                self.send_notification(
                    subject="Kubernetes Deployment Creation Failed",
                    message=f"Kubernetes deployment creation failed with the following error:\n\n{e}"
                )
                raise e

    def update_deployment(self):
        """
        Updates an existing Kubernetes Deployment.
        """
        logger.info("Updating Kubernetes Deployment.")
        try:
            deployment = self._build_deployment_object()
            response = self.apps_v1_api.patch_namespaced_deployment(
                name=self.kubernetes_config['deployment_name'],
                namespace=self.kubernetes_config['namespace'],
                body=deployment
            )
            logger.info(f"Deployment updated. Status: {response.metadata.name}")
        except ApiException as e:
            logger.error(f"Failed to update deployment: {e}")
            self.send_notification(
                subject="Kubernetes Deployment Update Failed",
                message=f"Kubernetes deployment update failed with the following error:\n\n{e}"
            )
            raise e

    def delete_deployment(self):
        """
        Deletes a Kubernetes Deployment.
        """
        logger.info("Deleting Kubernetes Deployment.")
        try:
            response = self.apps_v1_api.delete_namespaced_deployment(
                name=self.kubernetes_config['deployment_name'],
                namespace=self.kubernetes_config['namespace'],
                body=client.V1DeleteOptions(propagation_policy='Foreground')
            )
            logger.info(f"Deployment deleted. Status: {response.status}")
        except ApiException as e:
            logger.error(f"Failed to delete deployment: {e}")
            raise e

    def scale_deployment(self, replicas: int):
        """
        Scales the number of replicas in the Deployment.
        """
        logger.info(f"Scaling deployment to {replicas} replicas.")
        try:
            body = {'spec': {'replicas': replicas}}
            response = self.apps_v1_api.patch_namespaced_deployment_scale(
                name=self.kubernetes_config['deployment_name'],
                namespace=self.kubernetes_config['namespace'],
                body=body
            )
            logger.info(f"Deployment scaled to {replicas} replicas.")
        except ApiException as e:
            logger.error(f"Failed to scale deployment: {e}")
            self.send_notification(
                subject="Kubernetes Deployment Scaling Failed",
                message=f"Kubernetes deployment scaling failed with the following error:\n\n{e}"
            )
            raise e

    def create_service(self):
        """
        Creates a Kubernetes Service.
        """
        logger.info("Creating Kubernetes Service.")
        try:
            service = self._build_service_object()
            response = self.core_v1_api.create_namespaced_service(
                namespace=self.kubernetes_config['namespace'],
                body=service
            )
            logger.info(f"Service created. Name: {response.metadata.name}")
        except ApiException as e:
            if e.status == 409:
                logger.warning("Service already exists. Updating service.")
                self.update_service()
            else:
                logger.error(f"Failed to create service: {e}")
                self.send_notification(
                    subject="Kubernetes Service Creation Failed",
                    message=f"Kubernetes service creation failed with the following error:\n\n{e}"
                )
                raise e

    def update_service(self):
        """
        Updates an existing Kubernetes Service.
        """
        logger.info("Updating Kubernetes Service.")
        try:
            service = self._build_service_object()
            response = self.core_v1_api.patch_namespaced_service(
                name=self.kubernetes_config['service_name'],
                namespace=self.kubernetes_config['namespace'],
                body=service
            )
            logger.info(f"Service updated. Name: {response.metadata.name}")
        except ApiException as e:
            logger.error(f"Failed to update service: {e}")
            self.send_notification(
                subject="Kubernetes Service Update Failed",
                message=f"Kubernetes service update failed with the following error:\n\n{e}"
            )
            raise e

    def delete_service(self):
        """
        Deletes a Kubernetes Service.
        """
        logger.info("Deleting Kubernetes Service.")
        try:
            response = self.core_v1_api.delete_namespaced_service(
                name=self.kubernetes_config['service_name'],
                namespace=self.kubernetes_config['namespace']
            )
            logger.info(f"Service deleted. Status: {response.status}")
        except ApiException as e:
            logger.error(f"Failed to delete service: {e}")
            raise e

    def _build_deployment_object(self) -> client.V1Deployment:
        """
        Builds a V1Deployment object based on configurations.
        """
        container_env_vars = [
            client.V1EnvVar(name=key, value=value)
            for key, value in self.kubernetes_config['environment_variables'].items()
        ]

        container = client.V1Container(
            name=self.kubernetes_config['deployment_name'],
            image=self.kubernetes_config['container_image'],
            ports=[client.V1ContainerPort(container_port=self.kubernetes_config['container_port'])],
            env=container_env_vars
        )

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={'app': self.kubernetes_config['deployment_name']}),
            spec=client.V1PodSpec(containers=[container])
        )

        spec = client.V1DeploymentSpec(
            replicas=self.kubernetes_config['replicas'],
            selector={'matchLabels': {'app': self.kubernetes_config['deployment_name']}},
            template=template
        )

        deployment = client.V1Deployment(
            api_version='apps/v1',
            kind='Deployment',
            metadata=client.V1ObjectMeta(name=self.kubernetes_config['deployment_name']),
            spec=spec
        )

        return deployment

    def _build_service_object(self) -> client.V1Service:
        """
        Builds a V1Service object based on configurations.
        """
        ports = [client.V1ServicePort(
            port=self.kubernetes_config['container_port'],
            target_port=self.kubernetes_config['container_port']
        )]

        spec = client.V1ServiceSpec(
            selector={'app': self.kubernetes_config['deployment_name']},
            ports=ports,
            type='ClusterIP'
        )

        service = client.V1Service(
            api_version='v1',
            kind='Service',
            metadata=client.V1ObjectMeta(name=self.kubernetes_config['service_name']),
            spec=spec
        )

        return service

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.kubernetes_config['notification_recipients']
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
    Demonstrates example usage of the KubernetesManager class.
    """
    try:
        # Initialize KubernetesManager
        kubernetes_manager = KubernetesManager()

        # Create or update deployment
        kubernetes_manager.create_deployment()

        # Create or update service
        kubernetes_manager.create_service()

        # Scale deployment
        kubernetes_manager.scale_deployment(replicas=3)

        # Delete service
        # kubernetes_manager.delete_service()

        # Delete deployment
        # kubernetes_manager.delete_deployment()

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the Kubernetes manager example
    example_usage()
