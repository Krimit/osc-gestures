import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import time

from video_manager import VideoManager

from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 5056)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


#@markdown To better demonstrate the Hand Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

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
        self.time_of_last_callback = None
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


    def draw_landmarks_on_image(self, rgb_image, detection_result, start_time):
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

        delim = "\n"
        categories_to_print = delim.join(map(str, categories_and_scores))
        #print("akrim categories: \n" + categories_to_print)
        y0, dy = 20, 20#4
        for i, line in enumerate(categories_to_print.split('\n')):
            y = y0 + i*dy
            cv2.putText(annotated_image, line, (20, y ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2)

        #send face data via OSC
        row = map(str, categories_and_scores)
        client.send_message("/face", row)
      print("--- draw_landmarks_on_image %s ms ---" % ((time.time() * 1000) - (start_time * 1000)))
      self.time_of_last_callback = time.time()
      return annotated_image    


    def set_detector_result(self, result, output_image: mp.Image, timestamp_ms: int):
        print("--- loop time %s ms ---" % ((time.time() * 1000) - (timestamp_ms * 1000)))
        #print('hand landmarker result: {}'.format(result))
        self.detector_result = result
        self.mp_image = output_image
        self.time_of_last_callback = time.time()

    def result_is_ready(self):
        return self.detector_result is not None

    def annotate_image(self, mp_image: mp.Image):
        if not self.result_is_ready():
            return None
        img = mp_image
        if img is None:
            img = self.mp_image
        annotated_image = self.draw_landmarks_on_image(img.numpy_view(), self.detector_result, time.time())
        self.gesture_result = None
        return annotated_image

    def recognize_frame_async(self, is_enabled: bool, frame, timestamp_ms: int):
        if frame is None:
            return
            
        self.is_enabled = is_enabled      

        self.timestamp = timestamp_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.detector.detect_async(mp_image, self.timestamp)

    def init(self):
        self.timestamp = 0

        base_options = BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.set_detector_result,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        return self 

            
if __name__ == "__main__":
    with Mediapipe_FaceModule() as face_module:
        with VideoManager("Camera_1") as video_manager:
            while video_manager.is_open() and face_module.is_open():
                frame = video_manager.capture_frame(True)
                face_module.recognize_frame_async(True, frame)
                annotated_image = face_module.annotate_image()
                if annotated_image is None:
                    print("skipping frame")
                    video_manager.draw(frame)
                else:
                    video_manager.draw(annotated_image)         

