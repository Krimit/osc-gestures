import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_RIGHT_EYE, 
    FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL
)

import numpy as np

import time

from video_manager import VideoManager

import asyncio
import concurrent.futures

# Use a thread pool for the blocking OpenCV reads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


#IGNORED_PREDICTIONS = []
IGNORED_PREDICTIONS = [
    "_neutral",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFunnel",
    "noseSneerLeft",
    "noseSneerRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
]

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

BBOX_PADDING = 0.15
FULL_FACE_TOP_PADDING = 0.30  # Extra padding specifically for the top of the head

def _get_indices(connections):
    """Flattens mediapipe connection tuples into a unique list of landmark indices."""
    return list(set([idx for edge in connections for idx in edge]))

# Pre-calculate bbox targeted regions once.
FACE_REGIONS = {
    "face_mouth": _get_indices(FACEMESH_LIPS),
    # Combine eyes and brows into one upper-face region
    "face_eyes": _get_indices(FACEMESH_LEFT_EYE | FACEMESH_RIGHT_EYE | FACEMESH_LEFT_EYEBROW | FACEMESH_RIGHT_EYEBROW),
    "face_full": _get_indices(FACEMESH_FACE_OVAL)
}


class Mediapipe_FaceModule():
    """

    """

    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.detector_result = None
        self.video = None
        self.detector = None
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.time_of_last_callback = int(round(time.time() * 1000))
        self.measure_time = False
        self.init()

    def close(self):
        print("closing face model")
        self.detector.close()

    def __enter__(self):
        self.detector.__enter__()
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "timestamp: {}, detector: {}, detector_result: {}".format(self.timestamp, self.detector, self.detector_result)

    def is_open(self):
        return not self.quit

    def log_time(self, method_name: str, start_time: float):
        if self.measure_time:
            print("--- face.{} completed in {} ms ---".format(method_name, ((time.time() * 1000) - (start_time * 1000))))

    def _calculate_bounding_box(self, landmarks: list, region_indices: list[int], frame_width: int, frame_height: int, padding_pct: float, top_padding_pct: float = 0.0) -> list[int]:
        """Calculates a padded pixel bounding box for a specific subset of face landmarks."""
        x_coords = [landmarks[i].x for i in region_indices]
        y_coords = [landmarks[i].y for i in region_indices]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        w = x_max - x_min
        h = y_max - y_min
        
        pad_x = w * padding_pct
        pad_y = h * padding_pct
        
        # Calculate the asymmetrical top extension
        extra_top_pad = h * top_padding_pct
        
        # Clamp to 0.0 and 1.0 to prevent drawing outside the frame
        x_min = max(0.0, x_min - pad_x)
        y_min = max(0.0, y_min - pad_y - extra_top_pad) 
        x_max = min(1.0, x_max + pad_x)
        y_max = min(1.0, y_max + pad_y)
        
        return [
            int(x_min * frame_width),
            int(y_min * frame_height),
            int((x_max - x_min) * frame_width),
            int((y_max - y_min) * frame_height)
        ]


    @staticmethod
    def minify_floats(data, precision=4):
        if isinstance(data, list):
            return [round(x, precision) if isinstance(x, float) else x for x in data]
        elif isinstance(data, float):
            return round(data, precision)
        return data    

    def draw_landmarks_on_image(self, rgb_image, detection_result, start_time):
      if self.measure_time:  
          print("--- callback gap time %s ms ---" % ((start_time * 1000) - (self.time_of_last_callback * 1000)))
      #print("akrim detection_result: {}".format(detection_result.face_blendshapes))
      # for i in detection_result.face_blendshapes:
      #   stuff.add((i.score, i.category_name))
      #print("akrim stuff: {}".format(stuff))  
      face_landmarks_list = detection_result.face_landmarks
      annotated_image = np.copy(rgb_image)

      # Loop through the detected faces to visualize.
      for idx in range(len(face_landmarks_list)):
        categories_and_scores = [(i.category_name, i.score) for i in detection_result.face_blendshapes[idx]]
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_iris_connections_style())

        row = map(str, categories_and_scores)
      self.log_time("draw_landmarks_on_image", start_time)  
      #self.time_of_last_callback = time.time()
      return annotated_image    


    def stringify_detection(self, detection_result):  
        face_landmarks_list = detection_result.face_landmarks

        result = {}
        
        # 1. Grab the custom bboxes we attached in the callback
        bboxes = getattr(detection_result, 'custom_bboxes', {})

        # Loop through the detected faces
        for idx in range(len(face_landmarks_list)):
            categories_and_scores = [(i.category_name, self.minify_floats(i.score)) for i in detection_result.face_blendshapes[idx] if i.category_name not in IGNORED_PREDICTIONS]
            row = [x for t in categories_and_scores for x in t]
            result["face"] = row  
            
        # 2. Inject all regional bboxes into the final dictionary
        for key, bbox_array in bboxes.items():
            result[key] = bbox_array

        return result


    def set_detector_result(self, result, output_image: mp.Image, timestamp_ms: int):
        # Calculate regional bounding boxes
        bboxes = {}
        if result and result.face_landmarks:
            w, h = output_image.width, output_image.height
            # Assuming max 1 face
            landmarks = result.face_landmarks[0] 
            
            for region_name, indices in FACE_REGIONS.items():
                top_offset = FULL_FACE_TOP_PADDING if region_name == "face_full" else 0.0
                bboxes[f"bbox/{region_name}"] = self._calculate_bounding_box(
                    landmarks, indices, w, h, BBOX_PADDING, top_offset
                )
                
        if result:
            result.custom_bboxes = bboxes

        #print("--- loop time %s ms ---" % ((time.time() * 1000) - (timestamp_ms * 1000)))
        self.detector_result = result
        self.mp_image = output_image
        #if self.time_of_last_callback % 10 == 0:
        #    print("--- Face model result arrived. time since last result: {} ms ---".format((timestamp_ms - self.time_of_last_callback)))
        self.time_of_last_callback = int(round(time.time() * 1000))

    def result_is_ready(self):
        return self.detector_result is not None

    def consume_result(self):
        """Grabs the current result and resets the state for the next inference."""
        local_res = self.detector_result
        self.detector_result = None
        return local_res        

    def annotate_image(self, frame, result, camera_name):
        start_time = time.time()
        if result is None:
            return None, {}

        annotated_image = self.draw_landmarks_on_image(frame, result, time.time())
        result_dict = self.stringify_detection(result)

        label = f"{camera_name}"
        cv2.putText(annotated_image, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 3)
        return annotated_image, result_dict

    def recognize_frame_async(self, is_enabled: bool, frame, timestamp_ms: int):
        if frame is None:
            return
            
        self.is_enabled = is_enabled      

        self.timestamp = timestamp_ms
        self.frame = frame
        # this is only for GPU detection!
        #rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.detector.detect_async(mp_image, self.timestamp)       

    def init(self):
        self.timestamp = 0

        base_options = BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task', delegate=BaseOptions.Delegate.CPU)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.set_detector_result,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        return self 

async def main(face_module, video_manager):   
    while video_manager.is_open() and face_module.is_open():
        timestamp = int(time.time() * 1000)
        
        # 1. READ PHASE: Call the synchronous function directly for the test harness
        frame = video_manager.capture_frame()
        if frame is None:
            continue

        face_module.recognize_frame_async(True, frame, timestamp)
        if face_module.result_is_ready():
            # 1. Get the pristine production frame
            annotated_image, results_dict = face_module.annotate_image(face_module.frame, face_module.consume_result(), "testing")  
            
            # --- 2. ISOLATED TEST LOGIC: Draw BBoxes only for the local debug window ---
            for key, val in results_dict.items():
                if key.startswith("bbox/"):
                    x, y, w, h = val
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    label = key.replace("bbox/", "")
                    cv2.putText(annotated_image, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # ---------------------------------------------------------------------------

            # 3. Show the frame on your laptop screen
            video_manager.draw(annotated_image)
            # print("result values: {}".format(results_dict.values())) # Optional: comment out to reduce terminal spam
        else:
            # print("skipping annotation, model not ready")
            video_manager.draw(frame)
        
        await asyncio.sleep(0.001) # Yield control briefly to prevent locking the loop

            
if __name__ == "__main__":
    with Mediapipe_FaceModule() as face_module:
        with VideoManager("Camera_0") as video_manager:
            asyncio.run(main(face_module, video_manager))
      

