# src/modules/collaboration/staging.py

from collaboration_tools import CollaborationTools
from collaborative_workspace_dashboard import CollaborativeWorkspaceDashboard
from project_sharing_manager import ProjectSharingManager
from real_time_collaboration import RealTimeCollaboration
from secure_collaboration_protocol import SecureCollaborationProtocol
from secure_communication import SecureCommunication
from version_control import VersionControl
from video_voice_tools import VideoVoiceTools

# Expose these classes and functions to make them easily importable
__all__ = [
    "CollaborationTools",
    "CollaborativeWorkspaceDashboard",
    "ProjectSharingManager",
    "RealTimeCollaboration",
    "SecureCollaborationProtocol",
    "SecureCommunication",
    "VersionControl",
    "VideoVoiceTools",
]
