import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import time

from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 5056)

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult




#@markdown To better demonstrate the Hand Landmarker API, we have created a set of visualization tools that will be used in this colab. These will draw the landmarks on a detect person, as well as the expected connections between those markers.
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

class Mediapipe_FaceModule():
    """
    Gesture Categories: Unknown, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. 
    "_" means no hand detected, "Unknown" is used for hand detected but unknown gesture.
    """

    def __init__(self, camera_name: str):
        self.camera_name = camera_name
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
        print("closing hands video")
        self.video.release()
        self.detector.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def __enter__(self):
        self.detector.__enter__()
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "timestamp: {}, detector: {}, video: {}, detector_result: {}".format(self.timestamp, self.detector, self.video, self.detector_result)

    def is_open(self):
        return self.video.isOpened() and not self.quit   


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
        #print('hand landmarker result: {}'.format(result))
        self.detector_result = result

    def do_loop(self, is_enabled: bool, current_time):
        #print("do loop: {}".format(self))
        if not self.video.isOpened():
            print("video is closed, shutting down.")
            return

        # Capture frame-by-frame
        ret, frame = self.video.read()

        if not ret:
            print("Ignoring empty frame")
            return    
            
        self.is_enabled = is_enabled    
        if not(is_enabled):
            #print("loop - hands disabled")
            cv2.imshow('Show', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                print("Closing Camera Stream")
                self.quit = True
            return
        #else:
            #print("loop - hands enabled")    

        
        # flip so directions are more intuitive in the shown video. Only do this when using the table, not laptop camera.    
        #frame = cv2.flip(frame,-1)    

        self.timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.detector.detect_async(mp_image, self.timestamp)

        
        if (not (self.detector_result is None)):
            annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), self.detector_result, current_time)
            #cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Show',annotated_image)
        else:
            cv2.imshow('Show', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing Camera Stream")
            self.quit = True
            return    

        print("--- loop time %s ms ---" % ((time.time() * 1000) - (current_time * 1000)))

    def init(self):
        self.timestamp = 0
        camera_num_string = self.camera_name.split("_")[-1]
        try:
            camera_num = int(camera_num_string)
        except ValueError:
            print("Cannot convert to integer: {}. Defaulting to camera 0".format(camera_index))
            camera_num = 0
        self.video = cv2.VideoCapture(camera_num)
        print("debugging isOpened: {}".format(self.video.isOpened()))

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
    start_time = time.time()
    with Mediapipe_FaceModule("Camera_1") as face_module:
        face_module.time_of_last_callback = start_time
        while face_module.is_open():
            face_module.do_loop(True, time.time())

