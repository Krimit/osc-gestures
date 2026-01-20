import mediapipe as mp
import cv2
import time

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from video_manager import VideoManager
from metal_video_bridge import MetalVideoBridge
from syphon import SyphonMetalServer
import objc

import asyncio
import concurrent.futures

# Use a thread pool for the blocking OpenCV reads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
 
W, H = 1280, 720


MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (23, 26, 25) #(88, 205, 54) # vibrant green


BODY_DEPTH_LEVEL = 60
FACE_INTENSITY = 180
BLUR_AMOUNT = (15, 15)
DEPTH_EXPONENT = 0.3 # make higher for larger contrast
THICKNESS = 2 # to fill gaps between mesh lines

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
        self.frame = None
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

    # def draw_landmarks_on_image(self, rgb_image, frame, segmentation_result):
    #     mask = segmentation_result.confidence_masks[0].numpy_view()

    #     frame_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    #     # 3. Create a visual representation of the mask
    #     # Option A: Simple grayscale (multiply by factor if labels are small, e.g., 0 and 1)
    #     visual_mask = (mask * 255).astype(np.uint8)
        
    #     # Option B: Colored overlay (e.g., highlights the object in green)
    #     colored_mask = np.zeros_like(frame_bgr)
    #     colored_mask[mask > 0] = [0, 255, 0] # Green for detected segments
        
    #     # Blend the mask with the original frame
    #     overlay = cv2.addWeighted(frame_bgr, 0.6, colored_mask, 0.4, 0)

    #     #return visual_mask
    #     #cv2.imshow('Segmentation Mask', visual_mask)
    #     #cv2.imshow('Segmentation Overlay', overlay)

    #     # Apply mask to video immediately
    #     # Put the confidence map into the Alpha channel of the original video
        
    #     # Resize mask to match video frame (if needed, usually they match)
    #     # But normally MediaPipe output matches input size
        
    #     b, g, r = cv2.split(frame)
        
    #     # Scale confidence (0.0-1.0) to Alpha (0-255)
    #     alpha = (confidence_mask * 255).astype(np.uint8)
        
    #     # Create standard RGBA texture
    #     rgba_frame = cv2.merge([b, g, r, alpha])

    # Alpha version
    # def draw_landmarks_on_image(self, rgb_image, frame, segmentation_result):
    #     h, w, c = frame.shape
    #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     #timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    #     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #     confidence_mask = segmentation_result.confidence_masks[0].numpy_view()
    #     # Turn it into our "Base Depth Layer"
    #     # 1.0 confidence becomes BODY_DEPTH_LEVEL (e.g., 60)
    #     body_layer = (confidence_mask * BODY_DEPTH_LEVEL).astype(np.float32) 
        
    #     face_layer = np.zeros((h, w), dtype=np.float32)

    #     face_layer_blurred = cv2.GaussianBlur(face_layer, BLUR_AMOUNT, 0)
        
    #     combined_map = cv2.add(body_layer, face_layer_blurred)

    #     combined_map[confidence_mask < 0.1] = 0

    #     final_alpha = np.clip(combined_map, 0, 255).astype(np.uint8)
    #     b, g, r = cv2.split(frame)
    #     rgba_frame = cv2.merge([b, g, r, final_alpha])   

    #     return rgba_frame


    def draw_landmarks_on_image(self, rgb_image, frame, segmentation_result):
        # 1. Get raw mask
        confidence_mask = segmentation_result.confidence_masks[0].numpy_view()
        
        # 2. Scale mask to 0-255 uint8 (Full opacity for person)
        alpha_channel = (confidence_mask * 255).astype(np.uint8)
        
        # 3. Ensure frame is BGR (Standard for OpenCV)
        # If your original frame was already bright, don't modify BGR values
        b, g, r = cv2.split(frame)
        
        # 4. Merge - This creates a "Straight Alpha" image
        rgba_frame = cv2.merge([b, g, r, alpha_channel])
        
        return rgba_frame


    # # image version
    # def draw_landmarks_on_image(self, rgb_image, frame, segmentation_result):
    #     h, w, _ = frame.shape
        
    #     # 1. Get the confidence mask (values 0.0 to 1.0)
    #     # MediaPipe ImageSegmenter returns a list of masks; [0] is typically the person
    #     confidence_mask = segmentation_result.confidence_masks[0].numpy_view()

    #     # 2. Create a 3-channel boolean condition for the mask
    #     # Thresholding at 0.5 creates a clean binary cut
    #     condition = np.stack((confidence_mask,) * 3, axis=-1) > 0.5

    #     # 3. Define your background (e.g., solid black)
    #     # If you want a transparent background, keep using the RGBA merge logic
    #     bg_image = np.zeros(frame.shape, dtype=np.uint8)

    #     # 4. Extract the person: Where condition is True, use frame; else use bg_image
    #     extracted_person = np.where(condition, frame, bg_image)

    #     # Optional: If you still want the alpha channel for PNG saving
    #     # b, g, r = cv2.split(extracted_person)
    #     # alpha = (confidence_mask * 255).astype(np.uint8)
    #     # extracted_person = cv2.merge([b, g, r, alpha])

    #     return extracted_person


    def set_segmentation_result(self, result, output_image: mp.Image, timestamp_ms: int):
        print('segmentation result: {}'.format(result))
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

        self.frame = frame    

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


def publish_to_metal(bridge, syphon_server, frame):
    with objc.autorelease_pool():
        #Convert CPU Array -> GPU Texture
        mtl_texture = bridge.numpy_to_metal(frame)
        #Publish the Texture
        syphon_server.publish_frame_texture(mtl_texture)

async def main(segmentation_module, video_manager, bridge, syphon_server):  
    loop = asyncio.get_running_loop()  
    while video_manager.is_open() and segmentation_module.is_open():
        timestamp = int(time.time() * 1000)
        
        # 1. READ PHASE: Request frames from both cameras "simultaneously"
        # Because we use an executor, these run in parallel threads.
        frame = await loop.run_in_executor(executor, video_manager.capture_frame)

        # Wait to finish reading
        #await asyncio.gather(task)

        frame = video_manager.latest_frame
        if frame is None:
            continue

        segmentation_module.recognize_frame_async(True, frame, timestamp)
        while True:
            if segmentation_module.result_is_ready():
                annotated_image = segmentation_module.annotate_image(segmentation_module.mp_image, segmentation_module.frame)  
                video_manager.draw(annotated_image)
                publish_to_metal(bridge, syphon_server, annotated_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                break
            elif int(time.time() * 1000) % 10 == 0:
                print("skipping annotation, model not ready")
            await asyncio.sleep(0.001) 
        
        await asyncio.sleep(0.016) 


if __name__ == "__main__":
    with Mediapipe_SegmentationModule() as segmentation_module:
        with VideoManager("Camera_0") as video_manager:
            bridge = MetalVideoBridge(W, H)
            syphon_server = SyphonMetalServer("segmenter-test", device=bridge.device)
            asyncio.run(main(segmentation_module, video_manager, bridge, syphon_server))

# if __name__ == "__main__":
#     with Mediapipe_SegmentationModule() as segmentation_module:
#         with VideoManager("Camera_0") as video_manager:

#             while video_manager.is_open() and segmentation_module.is_open():
#                 timestamp = int(time.time() * 1000)
#                 frame = video_manager.capture_frame()
#                 if frame is None:
#                     continue
#                 segmentation_module.recognize_frame_async(True, frame, timestamp)
#                 if segmentation_module.result_is_ready():
#                     annotated_image = segmentation_module.annotate_image(segmentation_module.mp_image, frame)  
#                     video_manager.draw(annotated_image)
#                 else:
#                     print("skipping annotation, model not ready")
#                     #video_manager.draw(frame)
