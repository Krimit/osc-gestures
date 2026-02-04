from enum import Enum
import cv2

class CameraDirection(Enum):
    NORMAL = "NORMAL"
    FLIP_SIDE = "FLIP_SIDE"  # Horizontal Mirror (Standard Selfie)
    FLIP_TOP = "FLIP_TOP"    # Vertical Flip
    FLIP_BOTH = "FLIP_BOTH"  # 180 Degree Rotation

    def to_cv2_code(self):
        """Maps the enum to OpenCV flip codes."""
        match self:
            case CameraDirection.FLIP_BOTH:
                return -1
            case CameraDirection.FLIP_SIDE:
                return 1
            case CameraDirection.FLIP_TOP:
                return 0
            case _:
                return None