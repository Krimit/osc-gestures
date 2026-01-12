import mediapipe as mp
import cv2

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

import time

from video_manager import VideoManager


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

# parameters for the point cloud information
BODY_DEPTH_LEVEL = 60
FACE_INTENSITY = 180
BLUR_AMOUNT = (45, 45)

ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions


class Mediapipe_FaceSegModule():
    """

    """

    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.video = None
        self.detector_result = None
        self.detector = None
        self.segmentation_result = None
        self.segmenter = None
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



    # def paint_face_depth(image_shape, face_landmarks):
    #     """
    #     Takes existing landmarks and paints a depth map.
    #     image_shape: (height, width)
    #     face_landmarks: The list of landmarks from your EXISTING model
    #     """
    #     h, w = image_shape[:2]
        
    #     # Create black canvas (Float32 for smooth gradients)
    #     depth_canvas = np.zeros((h, w), dtype=np.float32)
        
    #     # CONFIG
    #     Z_SCALE = 150  # Pop out intensity
    #     DOT_SIZE = 20  # Size of each 'light'
        
    #     # If using standard MediaPipe "NormalizedLandmarkList"
    #     # We iterate directly over .landmark
    #     for lm in face_landmarks.landmark:
    #         x, y = int(lm.x * w), int(lm.y * h)
            
    #         # Invert Z: MediaPipe Z is negative for "close", positive for "far"
    #         # We want Close = Bright
    #         z = (lm.z * -1) + 0.1 
    #         brightness = max(0.0, z * Z_SCALE)
            
    #         # Draw the dot
    #         cv2.circle(depth_canvas, (x, y), DOT_SIZE, brightness, -1)

    #     # Blur to create surface
    #     depth_canvas = cv2.GaussianBlur(depth_canvas, (45, 45), 0)
        
    #     # Convert to 0-255 format
    #     return np.clip(depth_canvas, 0, 255).astype(np.uint8)

    def paint_face_depth(self, image_shape, detection_result):
        """
        Helper function: Converts YOUR existing landmarks into a depth texture.
        """
        h, w = image_shape[:2]
        depth_canvas = np.zeros((h, w), dtype=np.float32)
        
        for face in detection_result.face_landmarks:
            for lm in face:
                x, y = int(lm.x * w), int(lm.y * h)
                z_metric = (lm.z * -1) + 0.1
                brightness = max(0.0, z_metric * FACE_INTENSITY)

                # Draw large soft circles
                cv2.circle(depth_canvas, (x, y), 15, brightness, -1)

        return depth_canvas



    def create_alpha_depth(self, rgb_image, frame, segmentation_result, detection_result):
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        confidence_mask = segmentation_result.confidence_masks[0].numpy_view()
        # Turn it into our "Base Depth Layer"
        # 1.0 confidence becomes BODY_DEPTH_LEVEL (e.g., 60)
        body_layer = (confidence_mask * BODY_DEPTH_LEVEL).astype(np.float32) 
        
        face_layer = np.zeros((h, w), dtype=np.float32)

        if detection_result.face_landmarks:
            face_layer = self.paint_face_depth((h, w), detection_result)

        face_layer_blurred = cv2.GaussianBlur(face_layer, BLUR_AMOUNT, 0)
        
        combined_map = cv2.add(body_layer, face_layer_blurred)

        combined_map[confidence_mask < 0.1] = 0

        final_alpha = np.clip(combined_map, 0, 255).astype(np.uint8)
        b, g, r = cv2.split(frame)
        rgba_frame = cv2.merge([b, g, r, final_alpha])   

        return rgba_frame


    def draw_segmentation(self, rgb_image, frame, segmentation_result):
        mask = segmentation_result.confidence_masks[0].numpy_view()

        frame_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # 3. Create a visual representation of the mask
        # Option A: Simple grayscale (multiply by factor if labels are small, e.g., 0 and 1)
        visual_mask = (mask * 255).astype(np.uint8)
        
        # Option B: Colored overlay (e.g., highlights the object in green)
        colored_mask = np.zeros_like(frame_bgr)
        colored_mask[mask > 0] = [0, 255, 0] # Green for detected segments
        
        # Blend the mask with the original frame
        overlay = cv2.addWeighted(frame_bgr, 0.6, colored_mask, 0.4, 0)

        return visual_mask
        #cv2.imshow('Segmentation Mask', visual_mask)
        #cv2.imshow('Segmentation Overlay', overlay)

        # Apply mask to video immediately
        # Put the confidence map into the Alpha channel of the original video
        
        # Resize mask to match video frame (if needed, usually they match)
        # But normally MediaPipe output matches input size
        
        b, g, r = cv2.split(frame)
        
        # Scale confidence (0.0-1.0) to Alpha (0-255)
        alpha = (mask * 255).astype(np.uint8)
        
        # Create standard RGBA texture
        rgba_frame = cv2.merge([b, g, r, alpha])
        return visual_mask


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

        delim = "\n"
        categories_to_print = delim.join(map(str, categories_and_scores))
        #print("akrim categories: \n" + categories_to_print)
        y0, dy = 20, 20#4
        for i, line in enumerate(categories_to_print.split('\n')):
            y = y0 + i*dy
            cv2.putText(annotated_image, line, (20, y ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2)

        #send face data via OSC
        row = map(str, categories_and_scores)
      self.log_time("draw_landmarks_on_image", start_time)  
      #self.time_of_last_callback = time.time()
      return annotated_image    

    def stringify_detection(self, detection_result):  
      face_landmarks_list = detection_result.face_landmarks

      # Loop through the detected faces to visualize.
      result = {}
      for idx in range(len(face_landmarks_list)):
        categories_and_scores = [(i.category_name, i.score) for i in detection_result.face_blendshapes[idx] if i.category_name not in IGNORED_PREDICTIONS]
        row = [x for t in categories_and_scores for x in t]
        #print("SIZE {}, categories_and_scores {}".format(len(categories_and_scores), categories_and_scores))
        #print("row {}".format(row))

        # delim = "\n"
        # categories_to_print = delim.join(map(str, categories_and_scores))

        # #send face data via OSC
        # row = map(str, categories_to_print)
        # print("akrim row: {}".format(categories_to_print))
        result["face"] = row  
      return result

    def set_detector_result(self, result, output_image: mp.Image, timestamp_ms: int):
        #print("--- loop time %s ms ---" % ((time.time() * 1000) - (timestamp_ms * 1000)))
        print("--- Face model result arrived. timestamp_ms {}, time_of_last_callback {}, time since last result: {} ms ---".format(timestamp_ms, self.time_of_last_callback, (timestamp_ms - self.time_of_last_callback)))
        #print('hand landmarker result: {}'.format(result))
        self.detector_result = result
        self.mp_image = output_image
        self.time_of_last_callback = int(round(time.time() * 1000))

    def set_segmentation_result(self, result, output_image: mp.Image, timestamp_ms: int):
        print("--- Segmentation model result arrived. timestamp_ms {}, time_of_last_callback {}, time since last result: {} ms ---".format(timestamp_ms, self.time_of_last_callback, (timestamp_ms - self.time_of_last_callback)))
        self.segmentation_result = result
        self.mp_image = output_image

    def result_is_ready(self):
        return self.detector_result is not None and self.segmentation_result is not None

    def process_image(self, mp_image: mp.Image, frame):
        if not self.result_is_ready():
            print("results not ready!")
            return None
        annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), self.detector_result, time.time())
        alpha_depth_image = self.create_alpha_depth(mp_image.numpy_view(), frame, self.segmentation_result, self.detector_result)
        self.segmentation_result = None
        result_dict = self.stringify_detection(self.detector_result)
        return annotated_image, alpha_depth_image, result_dict        

    def recognize_frame_async(self, is_enabled: bool, frame, timestamp_ms: int):
        if frame is None:
            return
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)  

        self.is_enabled = is_enabled      
        self.timestamp = timestamp_ms
        self.segmenter.segment_async(mp_image, self.timestamp)           
        self.detector.detect_async(mp_image, self.timestamp)


    def init_face(self):
        self.timestamp = 0

        base_options = BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.set_detector_result,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        print("Finished initiating face detector Model.")
        return self 

    def init_segmenter(self):
        self.timestamp = 0

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path="models/selfie_segmentation_landscape.tflite"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            output_category_mask=False,
            output_confidence_masks=True, # Soft edges for point cloud
            result_callback=self.set_segmentation_result
        )

        self.segmenter = ImageSegmenter.create_from_options(options)
        print("Finished initiating Segmentation Model.")
        return self 

    def init(self):
        self.init_face()
        self.init_segmenter()
        return self

            
if __name__ == "__main__":
    with Mediapipe_FaceSegModule() as face_module:
        with VideoManager("Camera_0") as video_manager:
            while video_manager.is_open() and face_module.is_open():
                timestamp = int(time.time() * 1000)
                frame = video_manager.capture_frame(True)
                face_module.recognize_frame_async(True, frame, timestamp)
                if face_module.result_is_ready():
                    annotated_image, alpha_depth_image, results_dict = face_module.process_image(face_module.mp_image, frame)  
                    b, g, r, a = cv2.split(alpha_depth_image)
                    rgb_view = cv2.merge([b, g, r])
                    alpha_view = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
                    debug_image = np.hstack((rgb_view, alpha_view))
                    debug_image = np.hstack((annotated_image, debug_image))
                    video_manager.draw(debug_image)
                    print("result values: {}".format(results_dict.values()))
                else:
                    print("skipping annotation, model not ready")
                    #video_manager.draw(frame)
      

