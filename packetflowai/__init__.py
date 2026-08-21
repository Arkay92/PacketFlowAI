"""PacketFlowAI core package."""

from .features import FEATURE_SCHEMA_VERSION, PacketFeatures
from .hdc import ENCODER_SCHEMA_VERSION, HypervectorEncoder

__version__ = "4.0.0"

__all__ = [
    "ENCODER_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "HypervectorEncoder",
    "PacketFeatures",
    "__version__",
]
