from enum import Enum
from typing import NamedTuple, Optional

class ModelConfig(NamedTuple):
    detection_conf: float
    presence_conf: float
    tracking_conf: float

class ModelTarget(Enum):
# Hands Back (Top-down): Low detection, high tracking stability
    HANDS_BACK = ModelConfig(0.3, 0.3, 0.6)
    
    # Hands Front (Selfie/Webcam): Standard balanced values
    HANDS_FRONT = ModelConfig(0.5, 0.5, 0.5)
    
    FACE = ModelConfig(0.5, 0.5, 0.5)    
    BODY = ModelConfig(0.5, 0.5, 0.5)


    @property
    def config(self) -> ModelConfig:
        return self.value

    def __str__(self):
        return (
            f"[{self.name}] "
            f"Detection confidence: {self.value.detection_conf:.2f} | "
            f"Presence confidence: {self.value.presence_conf:.2f} | "
            f"Tracking confidence: {self.value.tracking_conf:.2f}"
        )