import mediapipe as mp
import cv2
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from video_manager import VideoManager


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (23, 26, 25) #(88, 205, 54) # vibrant green


BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode
#ImageSegmenterResult = mp.tasks.vision.ImageSegmenterResult


class Mediapipe_SegmentationModule():
    """
    """

    def __init__(self):
        self.mp_drawing = solutions.drawing_utils
        self.segmentation_result = None
        self.mp_image = None
        self.segmenter = None
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.init()

    def close(self):
        print("closing segmenter model")
        self.segmenter.close()

    def __enter__(self):
        self.segmenter.__enter__()
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "timestamp: {}, segmenter: {}, segmentation_result: {}".format(self.timestamp, self.segmenter, self.segmentation_result)

    def is_open(self):
        return not self.quit   

    def draw_landmarks_on_image(self, rgb_image, frame, segmentation_result):
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

        #return visual_mask
        #cv2.imshow('Segmentation Mask', visual_mask)
        #cv2.imshow('Segmentation Overlay', overlay)

        # Apply mask to video immediately
        # Put the confidence map into the Alpha channel of the original video
        
        # Resize mask to match video frame (if needed, usually they match)
        # But normally MediaPipe output matches input size
        
        b, g, r = cv2.split(frame)
        
        # Scale confidence (0.0-1.0) to Alpha (0-255)
        alpha = (confidence_mask * 255).astype(np.uint8)
        
        # Create standard RGBA texture
        rgba_frame = cv2.merge([b, g, r, alpha])


    def set_segmentation_result(self, result, output_image: mp.Image, timestamp_ms: int):
        #print('hand landmarker result: {}'.format(result))
        self.segmentation_result = result
        self.mp_image = output_image

    def result_is_ready(self):
        return self.segmentation_result is not None

    def annotate_image(self, mp_image: mp.Image, frame):
        if not self.result_is_ready():
            return None
        annotated_image = self.draw_landmarks_on_image(mp_image.numpy_view(), frame, self.segmentation_result)
        self.segmentation_result = None
        return annotated_image

    def recognize_frame_async(self, is_enabled: bool, frame, timestamp_ms: int):
        if frame is None:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)            
        self.is_enabled = is_enabled      
        self.timestamp = timestamp_ms
        self.segmenter.segment_async(mp_image, self.timestamp)


    def init(self):
        self.timestamp = 0
        gesture_model_path = "models/selfie_segmentation_landscape.tflite"

        options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=gesture_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            output_category_mask=False,
            output_confidence_masks=True, # Soft edges for point cloud
            result_callback=self.set_segmentation_result
        )

        self.segmenter = ImageSegmenter.create_from_options(options)
        print("Finished initiating Segmentation Model.")
        return self 

            
if __name__ == "__main__":
    with Mediapipe_SegmentationModule() as segmentation_module:
        with VideoManager("Camera_0") as video_manager:
            while video_manager.is_open() and segmentation_module.is_open():
                timestamp = int(time.time() * 1000)
                frame = video_manager.capture_frame(True)
                segmentation_module.recognize_frame_async(True, frame, timestamp)
                if segmentation_module.result_is_ready():
                    annotated_image = segmentation_module.annotate_image(segmentation_module.mp_image, frame)  
                    video_manager.draw(annotated_image)
                    #print(results_dict.values())
                else:
                    print("skipping annotation, model not ready")
                    #video_manager.draw(frame)
