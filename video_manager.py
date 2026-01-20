import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
import time

import asyncio
import concurrent.futures

# Use a thread pool for the blocking OpenCV reads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

class VideoManager():
    """
    """

    def __init__(self, camera_name: str, screen_xy: list = [0, 0]):
        self.camera_name = camera_name
        self.mp_drawing = solutions.drawing_utils
        self.video = None
        self.latest_frame = None
        self.timestamp = 0
        self.is_enabled = True
        self.quit = False
        self.screen_xy = screen_xy
        self.flip = False
        self.init()


    def close(self):
        print("closing video camera " + self.camera_name)
        self.video.release()
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
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("Closing Camera Stream")
        #     self.quit = True
        return 

    def package_for_draw(self, frame):
        return 

    def capture_frame(self):
        if not self.video.isOpened():
            print("video is closed, shutting down.")
            return None

        # Capture frame-by-frame, in thread
        # Note: no parens on "read", this is a method reference!
        ret, frame = self.video.read()

        if not ret:
            print("Ignoring empty frame")
            return None   

        self.latest_frame = frame
        
        # flip so directions are more intuitive in the shown video. Only do this when using the table, not laptop camera.    
        if self.flip:
            frame = cv2.flip(frame,-1)
        else:
            # flip right to left by default - for face case
            frame = cv2.flip(frame,1)

        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        return frame  

    async def capture_frame_async(self, is_enabled: bool):
        if not self.video.isOpened():
            print("video is closed, shutting down.")
            return None

        # Run the blocking "read()" in a separate thread, 
        # so the external main loop keeps spinning.
        loop = asyncio.get_running_loop()

        # Capture frame-by-frame, in thread
        # Note: no parens on "read", this is a method reference!
        ret, frame = await loop.run_in_executor(executor, self.video.read) 

        if not ret:
            print("Ignoring empty frame")
            return None   

        self.latest_frame = frame
        self.is_enabled = is_enabled                
        if not(is_enabled):
            #print("loop - hands disabled")
            #cv2.imshow('Show', frame)
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

        cv2.namedWindow(self.camera_name)
        print("setting camera {} to {}".format(self.camera_name, self.screen_xy))
        cv2.moveWindow(self.camera_name, self.screen_xy[0], self.screen_xy[1])
        while True:
            ret, frame = self.video.read()
            #self.draw(frame)
            if frame is None:
                print("frame is empty, waiting")
            if frame is not None:
                print("got a non-empty frame")
                break
        print("Video Camera Number {} isOpened: {}".format(camera_num, self.video.isOpened())) 
        return self 

async def main(video_manager):    
    while video_manager.is_open():
        # 1. READ PHASE: Request frames from both cameras "simultaneously"
        # Because we use an executor, these run in parallel threads.
        task = asyncio.create_task(video_manager.capture_frame(True))
        
        # Wait to finish reading
        await asyncio.gather(task)

        frame = video_manager.latest_frame

        video_manager.draw(frame)
        
        if frame is None:
            continue

        await asyncio.sleep(0)    

if __name__ == "__main__":
    with VideoManager("Camera_0") as video_manager:
        asyncio.run(main(video_manager))


