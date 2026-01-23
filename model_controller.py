import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
import time
from enum import Enum

from video_manager import VideoManager
from ghands_streaming import Mediapipe_HandsModule
from gface_streaming import Mediapipe_FaceModule
from gsegmenter_streaming import Mediapipe_SegmentationModule

import asyncio
import concurrent.futures
import objc

from method_timer import timeit_async

from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 5056)

from dataclasses import dataclass, field

# to reduce costs, we detect on smaller version of the frame
RESIZE_DIM = (640, 480)

class Detector(Enum):
    HANDS = 1
    FACE = 2
    HANDS_AND_FACE = 3
    SEGMENT = 4

@dataclass(frozen=True)  # Make instances immutable
class DetectedFrame:
    name : str
    original_frame: np.ndarray
    annotated_frame: np.ndarray
    detection_dict: dict

class ModelController():
    """
    """

    def __init__(self, video_manager: VideoManager, enabled_detector: Detector, executor, compute_segment: bool = False):
        self.video_manager = video_manager
        self.enabled_detector = enabled_detector
        self.hands_module = None
        self.face_module = None
        self.segment_module = None
        self.mp_drawing = solutions.drawing_utils
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.in_progress = False
        self.original_frame = None
        self.name = enabled_detector.name + "_" + video_manager.camera_name
        self.executor = executor
        self.compute_segment = compute_segment
        self.num_loops_waiting_for_results = 0
        self.init()

    def init(self):
        if self.enabled_detector == Detector.HANDS_AND_FACE:
            self.hands_module = Mediapipe_HandsModule()
            self.face_module = Mediapipe_FaceModule()
        elif self.enabled_detector == Detector.HANDS:
            self.hands_module = Mediapipe_HandsModule()
        elif self.enabled_detector == Detector.FACE:
            self.face_module = Mediapipe_FaceModule()
        elif self.enabled_detector == Detector.SEGMENT:
            self.segment_module = Mediapipe_SegmentationModule()  
        if self.compute_segment and not self.segment_module:
            self.segment_module = Mediapipe_SegmentationModule()     
        return self         

    def is_open(self):
        return self.video_manager.is_open() and \
             ((self.hands_module is not None and self.hands_module.is_open()) \
                or (self.face_module is not None and self.face_module.is_open())
                or (self.segment_module is not None and self.segment_module.is_open()))

    def close(self):
        print("closing deps")
        if self.hands_module is not None:
            self.hands_module.close()
        if self.face_module is not None:
            self.face_module.close()
        if self.segment_module is not None:
            self.segment_module.close()    

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "video_manager: {}, enabled_detector: {}".format(self.video_manager, self.enabled_detector) 

    @timeit_async
    async def _get_frame(self):
        """Offloads the blocking OpenCV read to a thread"""
        loop = asyncio.get_running_loop()
        frame = await loop.run_in_executor(self.executor, self.video_manager.capture_frame)
        return frame

    async def _get_annotated_frame(self, module, frame):
        loop = asyncio.get_running_loop()
        annotated_frame, results_dict = await loop.run_in_executor(
            self.executor, 
            module.annotate_image, 
            frame,
            self.name
        )
        return annotated_frame, results_dict

    async def get_frame_for_visualizing(self, model_frame):
        if self.segment_module:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
            self.executor, 
            self.segment_module.annotate_image, 
            self.segment_module.frame
        )
        return model_frame       

    def segment_can_continue(self):
        should_continue = True if not self.segment_module else self.segment_module.result_is_ready()
        return should_continue    
  

    def maybe_recognize_segment(self, frame, timestamp):
        if self.segment_module:
            self.segment_module.recognize_frame_async(True, frame, timestamp+1) 
        

    @timeit_async
    async def detect_hands_model(self):
        time_of_last_callback = self.timestamp
        if not self.is_open():
            return None
                
        if not self.in_progress:
            self.original_frame = await self._get_frame()
            self.timestamp = int(time.time() * 1000)
            small_frame = self.original_frame #small_frame = cv2.resize(self.original_frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
            self.hands_module.recognize_frame_async(True, small_frame, self.timestamp)
            self.in_progress = True
        if self.hands_module.result_is_ready():
            #print("Got result! Hands={}, segment={}".format(self.hands_module.result_is_ready(), self.segment_can_continue()))
            if self.num_loops_waiting_for_results > 2:
                print("hands result after {} loops".format(self.num_loops_waiting_for_results))
            annotated_image, results_dict = await self._get_annotated_frame(self.hands_module, self.hands_module.frame) #self.face_module.annotate_image(self.face_module.frame, self.name)
            self.in_progress = False
            self.num_loops_waiting_for_results = 0
            detection = DetectedFrame(self.name, None, annotated_image, results_dict)
            return detection
        else:
            #print("waiting for a result. Hands={}, segment={}".format(self.hands_module.result_is_ready(), self.segment_can_continue()))
            self.num_loops_waiting_for_results += 1
            return None    

    @timeit_async
    async def detect_face_model(self):
        time_of_last_callback = self.timestamp
        if not self.is_open():
            return None
                
        if not self.in_progress:
            self.original_frame = await self._get_frame()
            self.timestamp = int(time.time() * 1000)
            small_frame = self.original_frame #small_frame = cv2.resize(self.original_frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
            self.face_module.recognize_frame_async(True, small_frame, self.timestamp)
            self.in_progress = True
        if self.face_module.result_is_ready() and self.segment_can_continue():
            if self.num_loops_waiting_for_results > 2:
                print("face result after {} loops".format(self.num_loops_waiting_for_results))
            annotated_image, results_dict = await self._get_annotated_frame(self.face_module, self.face_module.frame) #self.face_module.annotate_image(self.face_module.frame, self.name)
            self.in_progress = False
            self.num_loops_waiting_for_results = 0
            detection = DetectedFrame(self.name, None, annotated_image, results_dict)
            return detection
        else:
            self.num_loops_waiting_for_results += 1
            return None 

    @timeit_async
    async def detect_segment(self):
        time_of_last_callback = self.timestamp
        if not self.is_open():
            return None
                
        if not self.in_progress:
            self.original_frame = await self._get_frame()
            self.timestamp = int(time.time() * 1000)
            small_frame = self.original_frame #small_frame = cv2.resize(self.original_frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
            self.segment_module.recognize_frame_async(True, small_frame, self.timestamp) 
            self.in_progress = True
        if self.segment_module.result_is_ready():
            if self.num_loops_waiting_for_results > 2:
                print("segment result after {} loops".format(self.num_loops_waiting_for_results))
            visual_frame = await self.get_frame_for_visualizing(self.segment_module.frame)                
            self.in_progress = False
            self.num_loops_waiting_for_results = 0
            detection = DetectedFrame(self.name, visual_frame, None, None)
            return detection
        else:
            self.num_loops_waiting_for_results += 1
            return None           

    def detect_hands_and_face_models(self):
        self.timestamp = int(time.time() * 1000)
        if not self.is_open():
            return
        frame = self.video_manager.capture_frame()
        if not self.in_progress:
            self.hands_module.recognize_frame_async(True, frame, self.timestamp)
            self.face_module.recognize_frame_async(True, frame, self.timestamp)
            self.in_progress = True
        if self.hands_module.result_is_ready() and self.face_module.result_is_ready():
            self.in_progress = False
            annotated_image, hands_results_dict = self.hands_module.annotate_image(self.hands_module.mp_image)
            if annotated_image is not None:
                result_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=annotated_image)
                annotated_image, face_results_dict = self.face_module.annotate_image(result_mp_image)
                self.video_manager.draw(annotated_image)
                return hands_results_dict | face_results_dict # merge the prediction results
        elif self.hands_module.result_is_ready():
            annotated_image, hands_results_dict = self.hands_module.annotate_image(self.hands_module.mp_image)  
            self.video_manager.draw(annotated_image)
            return hands_results_dict
        elif self.face_module.result_is_ready():
            annotated_image, face_results_dict = self.face_module.annotate_image(self.face_module.mp_image)  
            self.video_manager.draw(annotated_image)
            return face_results_dict
        else:
            #print("skipping annotation, models not ready")
            self.video_manager.draw(frame)

    async def detect(self):
        match self.enabled_detector:
            case Detector.HANDS:
                detection = await self.detect_hands_model()
            case Detector.FACE:
                detection = await self.detect_face_model()
            case Detector.HANDS_AND_FACE:
                detection = await self.detect_hands_and_face_models()
            case Detector.SEGMENT:
                detection = await self.detect_segment()    
            case _:
                raise Exception("Uhandled detector: " + str(enabled_detector))
        return detection


async def main_loop(video_manager, model_controller):
            while (model_controller.is_open()):
                osc_messages = await model_controller.detect()
                for message in osc_messages:
                    if message is not None:
                        client.send_message("/detect", message)
                        print("[debug] model result {}", message)
                await asyncio.sleep(0.001) # Give the loop a breather
            
if __name__ == "__main__":
    with VideoManager("Camera_0") as video_manager:
        with ModelController(video_manager, Detector.FACE) as model_controller:
            asyncio.run(main_loop(video_manager, model_controller))