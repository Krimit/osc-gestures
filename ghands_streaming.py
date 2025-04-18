import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from orientation_calculator import OrientationCalculator

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

class Mediapipe_HandsModule():
    """
    Gesture Categories: Unknown, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. 
    "_" means no hand detected, "Unknown" is used for hand detected but unknown gesture.
    """

    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.mp_drawing = solutions.drawing_utils
        self.landmark_result = None
        self.gesture_result = None
        self.video = None
        self.landmarker = None
        self.recognizer = None
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.init()

    def close(self):
        print("closing hands video")
        self.video.release()
        self.landmarker.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def __enter__(self):
        self.landmarker.__enter__()
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "timestamp: {}, landmarker: {}, video: {}, gesture_result: {}, landmark_result: {}".format(self.timestamp, self.landmarker, self.video, self.gesture_result, self.landmark_result)

    def is_open(self):
        return self.video.isOpened() and not self.quit   

    def draw_landmarks_on_image(self, rgb_image, landmark_result, gesture_result):
      #print("akrim gestures: {}".format(gesture_result.gestures))
      
      # TODO: Can we get rid of the landmark model entirely?
      #hand_landmarks_list = landmark_result.hand_landmarks
      #handedness_list = landmark_result.handedness
      
      hand_landmarks_list = gesture_result.hand_landmarks
      hand_world_landmark_list = gesture_result.hand_world_landmarks
      handedness_list = gesture_result.handedness
      gestures_list = gesture_result.gestures

      annotated_image = np.copy(rgb_image)

      # Loop through the detected hands to visualize.
      for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        world_landmarks = hand_world_landmark_list[idx]
        handedness = handedness_list[idx]
        gesture = gestures_list[idx]
        hand_direction = OrientationCalculator.calc(world_landmarks)

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name, gesture[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
        #send hand data via OSC
        hand = handedness[0].category_name
        gesture_category = gesture[0].category_name
        print("akrim category name: {}".format(gesture_category))
        row = []
        row += gesture_category
        row += hand_direction
        for idx, landmark in enumerate(hand_landmarks_proto.landmark):
            row += [idx, landmark.x, landmark.y, landmark.z]
            #print("handedness: {}, index: {}, {}, {}, {}".format(hand, idx, landmark.x, landmark.y, landmark.z))
            #client.send_message("/hand", [hand, idx, landmark.x, landmark.y, landmark.z])
        #print("hand: {}, row: {}".format(hand.lower(), row))
        client.send_message("/hand_" + hand.lower(), row)
      return annotated_image


    def set_landmark_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
        self.landmark_result = result
    
    def set_gesture_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
        self.gesture_result = result

    def do_loop(self, is_enabled: bool):
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing Camera Stream")
                self.quit = True
            return
        #else:
            #print("loop - hands enabled")    

        
        # flip so directions are more intuitive in the shown video. Only do this when using the table, not laptop camera.    
        #frame = cv2.flip(frame,-1)    

        self.timestamp += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, self.timestamp)
        self.recognizer.recognize_async(mp_image, self.timestamp)

        
        if (not (self.gesture_result is None)):
            annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), self.landmark_result, self.gesture_result)
            #cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.imshow('Show',annotated_image)
        else:
            cv2.imshow('Show', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing Camera Stream")
            self.quit = True
            return    

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

        gesture_model_path = "gesture_recognizer_hollow1.task"
        pretrain_model_path = "gesture_recognizer.task"
        landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence= 0.5,
            min_hand_presence_confidence= 0.5,
            min_tracking_confidence = 0.5,
            result_callback=self.set_landmark_result)
        self.landmarker = HandLandmarker.create_from_options(landmarker_options)

        gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=gesture_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.set_gesture_result)
        self.recognizer = GestureRecognizer.create_from_options(gesture_options)


        return self 

            
if __name__ == "__main__":
    with Mediapipe_HandsModule("Camera_1") as hands_module:
        while hands_module.is_open():
            hands_module.do_loop(True)

