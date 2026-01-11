import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
import time
from metal_video_bridge import MetalVideoBridge
from syphon import SyphonMetalServer

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

W, H = 1280, 720
SEND_TO_TD = True

class VideoManager():
    """
    """

    def __init__(self, camera_name: str, screen_xy: list = [0, 0]):
        self.camera_name = camera_name
        self.mp_drawing = solutions.drawing_utils
        self.video = None
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.screen_xy = screen_xy
        self.flip = False
        if SEND_TO_TD:
            self.bridge = MetalVideoBridge(W, H)
            self.syphon_server = SyphonMetalServer("HollowManVideo_" + camera_name, device=self.bridge.device)
        self.init()


    def close(self):
        print("closing video camera " + self.camera_name)
        self.video.release()
        if self.syphon_server:
            self.syphon_server.stop()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def __str__(self):
        return "video: {}".format(self.video)

    def is_open(self):
        return self.video.isOpened() and not self.quit   

    def set_flip(self, flip: bool):
        self.flip = flip

    def draw(self, frame):
        if frame is None:
            return
        cv2.imshow(self.camera_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing Camera Stream")
            self.quit = True
        return 

    def capture_frame(self, is_enabled: bool):
        if not self.video.isOpened():
            print("video is closed, shutting down.")
            return None

        # Capture frame-by-frame
        ret, frame = self.video.read()

        if not ret:
            print("Ignoring empty frame")
            return None   

        self.is_enabled = is_enabled                
        if not(is_enabled):
            #print("loop - hands disabled")
            cv2.imshow('Show', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Closing Camera Stream")
                self.quit = True
            return None 
        
        # flip so directions are more intuitive in the shown video. Only do this when using the table, not laptop camera.    
        # TODO(akrim): need input to flip or not based on the usecase. Do that in max when assigning the camera.
        if self.flip:
            frame = cv2.flip(frame,-1)
        else:
            # flip right to left by default - for face case
            frame = cv2.flip(frame,1)

        if SEND_TO_TD:
            # 1. Syphon/Metal requires 4 channels (Alpha). 
            # OpenCV defaults to BGR. We convert to BGRA.
            frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        
            # 2. Convert CPU Array -> GPU Texture
            mtl_texture = self.bridge.numpy_to_metal(frame_bgra)
        
            # 3. Publish the Texture
            self.syphon_server.publish_frame_texture(mtl_texture)

        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return frame       

    def init(self):
        print("init {}".format(self.camera_name))
        camera_num_string = self.camera_name.split("_")[-1]
        try:
            camera_num = int(camera_num_string)
        except ValueError:
            print("Cannot convert to integer: {}. Defaulting to camera 0".format(camera_index))
            camera_num = 0
        self.video = cv2.VideoCapture(camera_num)
        if SEND_TO_TD:
            # Set explicit resolution (important for the texture descriptor to match)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        cv2.namedWindow(self.camera_name)
        print("setting camera {} to {}".format(self.camera_name, self.screen_xy))
        cv2.moveWindow(self.camera_name, self.screen_xy[0], self.screen_xy[1])
        while True:
            frame = self.capture_frame(True)
            self.draw(frame)
            if frame is None:
                print("frame is empty, waiting")
            if frame is not None:
                print("got a non-empty frame")
                break
        print("Video Camera Number {} isOpened: {}".format(camera_num, self.video.isOpened())) 
        return self 

            
if __name__ == "__main__":
    with VideoManager("Camera_1") as video_manager:
        while video_manager.is_open():
            frame = video_manager.capture_frame(True)
            video_manager.draw(frame)


