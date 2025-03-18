import cv2

import numpy as np

from pythonosc.udp_client import SimpleUDPClient


client = SimpleUDPClient("127.0.0.1", 5056)

class CameraSetup():
    def __init__(self):
        self.camera_indexes = []
        self.videos = []
        self.quit = False


    def close(self):
        print("closing videos")
        for video in self.videos:
            video.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def __enter__(self):
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()

    def is_open(self):
        if self.quit:
            return False	
        video_opened = False
        for video in self.videos:
            if not(video.isOpened()):
                return False
            else:
                video_opened = True 
        return video_opened        

    def find_camera_indexes(self):
        # checks the first 10 indexes.
        index = 0
        arr = []
        while index < 10:
            cap = cv2.VideoCapture(index)
                
            if cap.read()[0]:
                arr.append(index)
            cap.release()
            index += 1

        print("camera indexes: {}".format(arr))    
        return arr

    def handle_video(self, index: int, video: cv2.VideoCapture):
    	# Capture frame-by-frame
        ret, frame = video.read()
        if not ret:
            print("Ignoring empty frame")
            return

        # f = frame.flatten()
        # print("%%%%%%%%")
        # print(str(np.shape(frame)))
        # print(str(np.shape(f)))
        # print("%%%%%%%%")
        # row = f.tolist()
        # print("akrim debugging numpy. Type: {}, data: {}".format(type(f), f))
        #client.send_message("/video_" + str(index), row)
        cv2.imshow('Show' + str(index), frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing Camera Stream")
            self.quit = True

    def do_loop(self, is_enabled: bool):
        if not self.is_open() and not self.quit:
            print("videos are closed, shutting down.")
            return  
        for index, video in enumerate(self.videos):
            self.handle_video(index, video)      

    def start_all_videos(self):
        self.camera_indexes = self.find_camera_indexes()
        self.videos = [cv2.VideoCapture(k) for k in self.camera_indexes]
        print("type: {}".format(type(self.videos[0])))
        return self 

            
if __name__ == "__main__":
    with CameraSetup() as camera_setup:
        camera_setup.start_all_videos()
        while camera_setup.is_open():
            camera_setup.do_loop(True)