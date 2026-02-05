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


    def draw_segment_as_alpha(self, frame, segmentation_result):
        # 1. Get raw mask
        confidence_mask = segmentation_result.confidence_masks[0].numpy_view()
        
        # 2. Scale mask to 0-255 uint8
        alpha_channel = (confidence_mask * 255).astype(np.uint8)
        
        # 3. GET FRAME DIMENSIONS
        # frame.shape is (height, width, channels)
        h, w = frame.shape[:2]
        
        # 4. RESIZE ALPHA TO MATCH FRAME
        # cv2.resize expects (width, height)
        alpha_resized = cv2.resize(alpha_channel, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 5. Ensure frame is BGR and split
        # If the frame already has 3 channels, we split them
        b, g, r = cv2.split(frame)
        
        # 6. Merge with the resized alpha
        rgba_frame = cv2.merge([b, g, r, alpha_resized])
        
        return rgba_frame


    def set_segmentation_result(self, result, output_image: mp.Image, timestamp_ms: int):
        self.segmentation_result = result
        self.mp_image = output_image

    def result_is_ready(self):
        return self.segmentation_result is not None

    def annotate_image(self, frame):
        if not self.result_is_ready():
            return None
        annotated_image = self.draw_segment_as_alpha(frame, self.segmentation_result)
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


def publish_to_metal(bridge, frame):
    with objc.autorelease_pool():        
        bridge.publish_to_metal(frame)

async def main(segmentation_module, video_manager, bridge):  
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
                annotated_image = segmentation_module.annotate_image(segmentation_module.frame)  
                video_manager.draw(annotated_image)
                publish_to_metal(bridge, annotated_image)
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
            bridge = MetalVideoBridge(W, H, "segmenter-test")
            asyncio.run(main(segmentation_module, video_manager, bridge))
