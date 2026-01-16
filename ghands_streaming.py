import mediapipe as mp
import cv2
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from orientation_calculator import OrientationCalculator
from video_manager import VideoManager


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (23, 26, 25) #(88, 205, 54) # vibrant green

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult



class Mediapipe_HandsModule():
    """
    Gesture Categories: Unknown, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. 
    "_" means no hand detected, "Unknown" is used for hand detected but unknown gesture.
    """

    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.landmark_result = None
        self.gesture_result = None
        self.mp_image = None
        self.landmarker = None
        self.recognizer = None
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.init()

    def close(self):
        print("closing hands model")
        self.landmarker.close()
        self.recognizer.close()

    def __enter__(self):
        self.landmarker.__enter__()
        self.recognizer.__enter__()
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "timestamp: {}, landmarker: {}, gesture_result: {}, landmark_result: {}".format(self.timestamp, self.landmarker, self.gesture_result, self.landmark_result)

    def is_open(self):
        return not self.quit   

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
        
      return annotated_image

    def stringify_detection(self, gesture_result):      
      hand_landmarks_list = gesture_result.hand_landmarks
      hand_world_landmark_list = gesture_result.hand_world_landmarks
      handedness_list = gesture_result.handedness
      gestures_list = gesture_result.gestures

      result = {}
      # Loop through the detected hands.
      #print("akrim debugging: {}".format(gesture_result))
      #print("handedness_list: {}. len: {}".format(handedness_list, len(handedness_list)))
      #return "akrim"
      for idx in range(len(handedness_list)):
        hand_landmarks = hand_landmarks_list[idx]
        world_landmarks = hand_world_landmark_list[idx]
        handedness = handedness_list[idx]
        gesture = gestures_list[idx]
        hand_direction = OrientationCalculator.calc(world_landmarks)

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        
        #send hand data via OSC
        hand = handedness[0].category_name.lower()
        gesture_category = gesture[0].category_name
        if not gesture_category:
            gesture_category = "_" # signifies that no gesture is detected. This works better in MaxMSP than trying to find an empty string.
        #print("akrim category name: {}".format(gesture_category))
        row = []
        row.append(hand)
        row.append(gesture_category)
        row.append(hand_direction)
        for i, landmark in enumerate(hand_landmarks_proto.landmark):
            row.extend([i, landmark.x, landmark.y, landmark.z])
            #print("handedness: {}, index: {}, {}, {}, {}".format(hand, i, landmark.x, landmark.y, landmark.z))
        #print("hand: {}, row: {}".format(hand.lower(), row))
        #print("akrim idx: {}, prev result: {}".format(idx, result))
        result["hand/" + hand] = row
      #print("akrim hands result: {}".format(result))
      return result


    def set_landmark_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
        self.landmark_result = result
        self.mp_image = output_image
    
    def set_gesture_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
        self.gesture_result = result
        self.mp_image = output_image

    def result_is_ready(self):
        return self.gesture_result is not None

    def annotate_image(self, frame):
        if not self.result_is_ready():
            return None
        annotated_image = self.draw_landmarks_on_image(frame, self.landmark_result, self.gesture_result)
        #print("akrim type of hands annotated_image {}".format(type(annotated_image)))
        result_dict = self.stringify_detection(self.gesture_result)
        self.gesture_result = None
        self.landmark_result = None
        return annotated_image, result_dict

    def recognize_frame_async(self, is_enabled: bool, frame, timestamp_ms: int):
        if frame is None:
            return
        
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgba_frame)
        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)    
        self.is_enabled = is_enabled      
        self.timestamp = timestamp_ms
        self.frame = frame
        self.landmarker.detect_async(mp_image, self.timestamp)
        self.recognizer.recognize_async(mp_image, self.timestamp)


    def init(self):
        self.timestamp = 0
        gesture_model_path = "models/model_training_4/gesture_recognizer.task"
        # pretrain_model_path = "gesture_recognizer.task"
        landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task', delegate=BaseOptions.Delegate.GPU),
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
        print("Finished initiating Hands Model.")
        return self 

            
if __name__ == "__main__":
    with Mediapipe_HandsModule() as hands_module:
        with VideoManager("Camera_1") as video_manager:
            while video_manager.is_open() and hands_module.is_open():
                timestamp = int(time.time() * 1000)
                frame = video_manager.capture_frame(True)
                hands_module.recognize_frame_async(True, frame, timestamp)
                if hands_module.result_is_ready():
                    annotated_image, results_dict = hands_module.annotate_image(hands_module.frame)  
                    video_manager.draw(annotated_image)
                    print(results_dict.values())
                else:
                    print("skipping annotation, model not ready")
                    video_manager.draw(frame)
