import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
import time
from enum import Enum

from video_manager import VideoManager
from ghands_streaming import Mediapipe_HandsModule
from gface_streaming import Mediapipe_FaceModule

from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 5056)


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


class Detector(Enum):
    HANDS = 1
    FACE = 2
    HANDS_AND_FACE = 3

class ModelController():
    """
    """

    def __init__(self, video_manager: VideoManager, enabled_detector: Detector):
        self.video_manager = video_manager
        self.enabled_detector = enabled_detector
        self.hands_module = None
        self.face_module = None
        self.mp_drawing = solutions.drawing_utils
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.init()

    def close(self):
        print("closing deps")
        if self.hands_module is not None:
            self.hands_module.close()
        if self.face_module is not None:
            self.face_module.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "video_manager: {}, hands_module: {}, face_module {}".format(self.video_manager, self.hands_module, self.face_module) 

    def is_open(self):
        return self.video_manager.is_open() and \
             ((self.hands_module is not None and self.hands_module.is_open()) \
                or (self.face_module is not None and self.face_module.is_open()))
    

    def init(self):
        if self.enabled_detector == Detector.HANDS_AND_FACE:
            self.hands_module = Mediapipe_HandsModule()
            self.face_module = Mediapipe_FaceModule()
        elif self.enabled_detector == Detector.HANDS:
            self.hands_module = Mediapipe_HandsModule()
        elif self.enabled_detector == Detector.FACE:
            self.face_module = Mediapipe_FaceModule()
        frame = self.video_manager.capture_frame(True)           
        return self 

    def detect_hands_model(self):
        timestamp = int(time.time() * 1000)
        if not self.is_open():
            return
        frame = self.video_manager.capture_frame(True)
        self.hands_module.recognize_frame_async(True, frame, timestamp)
        if self.hands_module.result_is_ready():
            annotated_image, results_dict = self.hands_module.annotate_image(self.hands_module.mp_image)  
            self.video_manager.draw(annotated_image)
            return results_dict.values()
        else:
            print("skipping annotation, model not ready")
            self.video_manager.draw(frame)
            return []    

    def detect_face_model(self):
        timestamp = int(time.time() * 1000)
        if not self.is_open():
            return
        frame = self.video_manager.capture_frame(True)
        self.face_module.recognize_frame_async(True, frame, timestamp)
        if self.face_module.result_is_ready():
            annotated_image = self.face_module.annotate_image(self.face_module.mp_image)  
            self.video_manager.draw(annotated_image)
        else:
            print("skipping annotation, model not ready")
            self.video_manager.draw(frame)

    def detect_hands_and_face_models(self):
        timestamp = int(time.time() * 1000)
        if not self.is_open():
            return
        frame = self.video_manager.capture_frame(True)
        self.hands_module.recognize_frame_async(True, frame, timestamp)
        self.face_module.recognize_frame_async(True, frame, timestamp)
        if self.hands_module.result_is_ready() and self.face_module.result_is_ready():
            annotated_image = self.hands_module.annotate_image(self.hands_module.mp_image)
            if annotated_image is not None:
                result_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=annotated_image)
                annotated_image = self.face_module.annotate_image(result_mp_image)
                self.video_manager.draw(annotated_image)
        elif self.hands_module.result_is_ready():
            annotated_image = self.hands_module.annotate_image(self.hands_module.mp_image)  
            self.video_manager.draw(annotated_image)
        elif self.face_module.result_is_ready():
            annotated_image = self.face_module.annotate_image(self.face_module.mp_image)  
            self.video_manager.draw(annotated_image)
        else:
            print("skipping annotation, models not ready")
            self.video_manager.draw(frame)

    def detect(self):
        match self.enabled_detector:
            case Detector.HANDS:
                return self.detect_hands_model()
            case Detector.FACE:
                return self.detect_face_model()
            case Detector.HANDS_AND_FACE:
                return self.detect_hands_and_face_models()
            case _:
                raise Exception("Uhandled detector: " + str(enabled_detector))            

            
if __name__ == "__main__":
    with VideoManager("Camera_1") as video_manager:
        with ModelController(video_manager, Detector.HANDS) as model_controller:
            while (model_controller.is_open()):
                osc_messages = model_controller.detect()
                for message in osc_messages:
                    if message is not None:
                        client.send_message("/detect", message)
