import mediapipe as mp
import cv2
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from orientation_calculator import OrientationCalculator
from video_manager import VideoManager
from model_target import ModelTarget


MARGIN = 10  # pixels
FONT_SIZE = 3
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (23, 26, 25) #(88, 205, 54) # vibrant green

# bounding box padding
BBOX_PADDING = 0.15  # Configurable buffer: adds 15% padding around the hand

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult



class Mediapipe_HandsModule():
    """
    Gesture Categories: Unknown, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. 
    "_" means no hand detected, "Unknown" is used for hand detected but unknown gesture.
    """

    def __init__(self, model_target: ModelTarget = ModelTarget.HANDS_FRONT, invert_handedness: bool = False):
        self.mp_drawing = solutions.drawing_utils
        self.gesture_result = None
        self.mp_image = None
        self.recognizer = None
        self.timestamp = 0
        self.time_of_last_callback = int(round(time.time() * 1000))
        self.is_enabled = True
        self.quit = False
        self.model_target = model_target
        self.invert_handedness = invert_handedness
        self.init()

    def close(self):
        print("closing hands model")
        self.recognizer.close()

    def __enter__(self):
        self.recognizer.__enter__()
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "timestamp: {}, gesture_result: {}, landmark_result: {}".format(self.timestamp, self.gesture_result)

    def is_open(self):
        return not self.quit   

    @staticmethod
    def minify_floats(data, precision=4):
        if isinstance(data, list):
            return [round(x, precision) if isinstance(x, float) else x for x in data]
        elif isinstance(data, float):
            return round(data, precision)
        return data    

    def _calculate_bounding_box(self, landmarks, frame_width, frame_height, padding_pct):
        """Calculates a padded pixel bounding box from normalized landmarks."""
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        w = x_max - x_min
        h = y_max - y_min
        
        pad_x = w * padding_pct
        pad_y = h * padding_pct
        
        x_min = max(0.0, x_min - pad_x)
        y_min = max(0.0, y_min - pad_y)
        x_max = min(1.0, x_max + pad_x)
        y_max = min(1.0, y_max + pad_y)
        
        return [
            int(x_min * frame_width),
            int(y_min * frame_height),
            int((x_max - x_min) * frame_width),
            int((y_max - y_min) * frame_height)
        ]

    def draw_landmarks_on_image(self, rgb_image, gesture_result):
        hand_landmarks_list = gesture_result.hand_landmarks
        hand_world_landmark_list = gesture_result.hand_world_landmarks
        handedness_list = gesture_result.handedness
        gestures_list = gesture_result.gestures

        bboxes = getattr(gesture_result, 'custom_bboxes', {})

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

            # Utilize the pre-calculated bounding box for both the debug drawing AND text placement
            hand_label = handedness[0].category_name.lower()
            if f"hand/{hand_label}" in bboxes:
                x, y, w, h = bboxes[f"hand/{hand_label}"]

                # Draw the debug bounding box (Thin Cyan)
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 255, 0), 1)

                # Use the top-left corner of the bbox for text
                text_x, text_y = x, max(0, y - MARGIN)
            else:
                text_x, text_y = 50, 50 # Fallback

            # Draw handedness/gesture text
            cv2.putText(annotated_image, f"{hand_label}, {gesture[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image

    def stringify_detection(self, gesture_result):      
      hand_landmarks_list = gesture_result.hand_landmarks
      hand_world_landmark_list = gesture_result.hand_world_landmarks
      handedness_list = gesture_result.handedness
      gestures_list = gesture_result.gestures

      result = {}

      bboxes = getattr(gesture_result, 'custom_bboxes', {})
      
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
            row.extend([i, self.minify_floats(landmark.x), self.minify_floats(landmark.y), self.minify_floats(landmark.z)])
            #print("handedness: {}, index: {}, {}, {}, {}".format(hand, i, landmark.x, landmark.y, landmark.z))
        #print("hand: {}, row: {}".format(hand.lower(), row))
        #print("akrim idx: {}, prev result: {}".format(idx, result))
        result["hand/" + hand] = row

        # 4. INJECT THE BBOX INTO THE DICTIONARY
        if f"hand/{hand}" in bboxes:
            result["bbox/" + hand] = bboxes[f"hand/{hand}"]
      #print("akrim hands result: {}".format(result))
      return result
    
    def set_gesture_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        if self.invert_handedness and result is not None and result.handedness:
            for categories in result.handedness:
                # 'categories' is a list of classifications for a single hand (usually length 1)
                for category in categories:
                    if category.category_name == "Left":
                        category.category_name = "Right"
                        category.display_name = "Right" 
                    elif category.category_name == "Right":
                        category.category_name = "Left"
                        category.display_name = "Left"
        bboxes = {}
        if result and result.handedness:
            w, h = output_image.width, output_image.height
            for idx in range(len(result.handedness)):
                hand_label = result.handedness[idx][0].category_name.lower()
                bboxes[f"hand/{hand_label}"] = self._calculate_bounding_box(result.hand_landmarks[idx], w, h, BBOX_PADDING)
        if result:
            result.custom_bboxes = bboxes        

        self.gesture_result = result
        self.mp_image = output_image
        #if self.time_of_last_callback % 10 == 0:
        #    print("--- Hand Gesture model result arrived. time since last result: {} ms ---".format((timestamp_ms - self.time_of_last_callback)))
        self.time_of_last_callback = int(round(time.time() * 1000))


    def result_is_ready(self):
        return self.gesture_result is not None

    def consume_result(self):
        """Grabs the current result and resets the state for the next inference."""
        local_res = self.gesture_result
        self.gesture_result = None
        return local_res    

    def annotate_image(self, frame, result, camera_name):
        if result is None:
            return None, {}
        annotated_image = self.draw_landmarks_on_image(frame, result)
        result_dict = self.stringify_detection(result)
        
        # Add text overlay to the individual frame
        label = f"{camera_name}"
        cv2.putText(annotated_image, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 3)
        
        return annotated_image, result_dict

    def recognize_frame_async(self, is_enabled: bool, frame, timestamp_ms: int):
        if frame is None:
            return

        # if False:
        #     # Flip top to bottom (0) if using table/special mount. Would normally use -1 (both), but we already flipped in the camera directly.
        #     #frame = cv2.flip(frame, 0)   
        #     frame = cv2.flip(frame, -1)  
        #     print("ghands_streaming cv2.flip(frame, -1)")   
        
        # This is only for GPU detection
        #rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=rgb_frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)    
        self.is_enabled = is_enabled      
        self.timestamp = timestamp_ms
        self.frame = frame
        self.recognizer.recognize_async(mp_image, self.timestamp)


    def init(self):
        self.timestamp = 0
        gesture_model_path = "models/model_training_4/gesture_recognizer.task"
        # pretrain_model_path = "gesture_recognizer.task"
        
        print(f"initiating hands model with config: {self.model_target}")
        min_hand_detection_confidence = self.model_target.config.detection_conf
        min_hand_presence_confidence = self.model_target.config.presence_conf
        min_tracking_confidence = self.model_target.config.tracking_conf

        gesture_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=gesture_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self.set_gesture_result)
        self.recognizer = GestureRecognizer.create_from_options(gesture_options)
        print("Finished initiating Hands Model.")
        return self 

            
if __name__ == "__main__":
    with Mediapipe_HandsModule(model_target=ModelTarget.HANDS_BACK, invert_handedness=True) as hands_module:
        with VideoManager("Camera_1") as video_manager:
            while video_manager.is_open() and hands_module.is_open():
                timestamp = int(time.time() * 1000)
                frame = video_manager.capture_frame()
                # assuming sitting in front of camera
                frame = cv2.flip(frame, 0)

                hands_module.recognize_frame_async(True, frame, timestamp)
                if hands_module.result_is_ready():
                    annotated_image, results_dict = hands_module.annotate_image(hands_module.frame, hands_module.consume_result(), "testing")  
                    video_manager.draw(annotated_image)
                    print(results_dict.values())
                else:
                    print("skipping annotation, model not ready")
                    #video_manager.draw(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break    
