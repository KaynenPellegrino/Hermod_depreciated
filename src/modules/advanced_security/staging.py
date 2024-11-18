# src/modules/advanced_security/staging.py

from .behavioral_authentication import BehavioralAuthenticationManager, BehavioralAuthenticationModel, BehavioralProfile
from .emerging_threat_detector import ThreatEvent, ThreatDetectionModel, EmergingThreatDetector
from .quantum_resistant_algorithms import QuantumResistantCryptography

# Expose these classes to make them easily importable from this module
__all__ = [
    "BehavioralAuthenticationManager",
    "BehavioralAuthenticationModel",
    "BehavioralProfile",
    "ThreatEvent",
    "ThreatDetectionModel",
    "EmergingThreatDetector",
    "QuantumResistantCryptography",
]
