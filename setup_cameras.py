import cv2

import numpy as np

from video_manager import VideoManager

from pythonosc.udp_client import SimpleUDPClient

from model_controller import Detector



client = SimpleUDPClient("127.0.0.1", 5056)

class CameraSetup():
    def __init__(self):
        self.camera_indexes = []
        self.videos = []
        self.video_managers = {}
        self.names = []
        self.quit = False


    def close(self):
        print("closing videos")
        # for vm in video_managers:
        #     vm.close()
        # for video in self.videos:
        #     video.release()
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
        for video in self.video_managers.values():
            if not(video.is_open()):
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

    def handle_video(self, name: str, video_manager: VideoManager):
    	# Capture frame-by-frame
        frame = video_manager.capture_frame()
        video_manager.draw(frame)

        # ret, frame = video.read()
        # if not ret:
        #     print("Ignoring empty frame")
        #     return

        # camera_name = self.names[index]
        # cv2.imshow(camera_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Closing Camera Stream")
            self.quit = True

    def do_loop(self, is_enabled: bool):
        if not self.is_open() and self.quit:
            print("videos are closed, shutting down.")
            return  
        for name, video_manager in self.video_managers.items():
            self.handle_video(name, video_manager)

    def stop_unused_cameras(self, cameras_to_use: list[str]):
        cameras_to_delete = [c for c in self.names if not c in cameras_to_use]
        for to_delete in cameras_to_delete:
            self.names.remove(to_delete)
            self.video_managers[to_delete].close()
            del self.video_managers[to_delete]

    def set_camera_orientation_by_model(self, camera_name_to_detector):
        for camera_name, detector in camera_name_to_detector.items():
            flip = False
            if detector == Detector.HANDS:
                flip = True
            self.video_managers[camera_name].set_flip(flip)


    def create_starting_locations(self, names: list):
        result = {}
        xy = [0, 0]
        for k in names:
            result[k] = xy
            xy = [xy[0] + 50, xy[1] + 50]
        print("video locations will be: {}".format(result))
        return result

    def start_all_videos(self):
        if len(self.video_managers) > 0:
            self.__init__()
        self.camera_indexes = self.find_camera_indexes()
        self.names = ["Camera_" + str(i) for i in self.camera_indexes]
        positions = self.create_starting_locations(self.names)
        self.video_managers = {k: VideoManager(k, positions[k]) for k in self.names}

        names = ["None"] + self.names
        
        print("sending camera names to Max: {}".format(names))
        print("managers: {}".format(self.video_managers))
        client.send_message("/camera_names", names)
        return self     

            
if __name__ == "__main__":
    with CameraSetup() as camera_setup:
        camera_setup.start_all_videos()
        while camera_setup.is_open():
            camera_setup.do_loop(True)