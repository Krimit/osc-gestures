import cv2
import threading
import time
import asyncio

class VideoManager:
    def __init__(self, camera_name: str, screen_xy: list = [0, 0], width=1280, height=720, target_fps=60):
        self.camera_name = camera_name
        self.screen_xy = screen_xy
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.flip = False  # Default flip state
        
        self.video = None
        self.latest_frame = None
        
        # Thread control
        self.stopped = False
        self._lock = threading.Lock()
        
        # Initialize camera and window
        self.init()
        
        # Start the background producer thread immediately
        self.thread = threading.Thread(target=self._update_loop, args=())
        self.thread.daemon = True
        self.thread.start()

    def init(self):
        print(f"init {self.camera_name}")
        camera_num_string = self.camera_name.split("_")[-1]
        try:
            camera_num = int(camera_num_string)
        except ValueError:
            print(f"Cannot convert to integer: {camera_num_string}. Defaulting to 0")
            camera_num = 0
            
        # 1. Open Camera
        self.video = cv2.VideoCapture(camera_num)
        
        # Set to target_fps if hardware supports it
        self.video.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # 2. Setup Window (Keep your existing window logic)
        cv2.namedWindow(self.camera_name)
        print(f"setting camera {self.camera_name} to {self.screen_xy}")
        cv2.moveWindow(self.camera_name, self.screen_xy[0], self.screen_xy[1])

        # 3. Warm up: Block until we get at least one valid frame
        # This ensures the rest of the app doesn't start with None
        while True:
            ret, frame = self.video.read()
            if ret and frame is not None:
                self.latest_frame = frame
                print("got a non-empty frame")
                break
            print("frame is empty, waiting...")
            time.sleep(0.1)
            
        print(f"Video Camera {camera_num} isOpened: {self.video.isOpened()}")

    def _update_loop(self):
        """
        Background Producer:
        Constantly grabs frames, flips them, and stores them.
        This runs completely independent of your main loop.
        """
        while not self.stopped:
            if not self.video.isOpened():
                break
                
            ret, frame = self.video.read()
            
            if not ret:
                # If stream is dead, wait a bit and try again to avoid CPU spin
                time.sleep(0.01)
                continue

            # Process the frame HERE (in the background) so main thread doesn't pay for it
            if self.flip:
                # Flip -1 (both) if using table/special mount
                frame = cv2.flip(frame, -1)
            else:
                # Default mirror flip (1)
                frame = cv2.flip(frame, 1)

            # Update shared memory safely
            with self._lock:
                self.latest_frame = frame

    def capture_frame(self):
        """
        Consumer:
        Returns the latest frame INSTANTLY.
        No waiting for hardware.
        """
        with self._lock:
            if self.latest_frame is None:
                return None
            # Return a COPY so your models can draw on it without messing up others
            return self.latest_frame.copy()

    def set_flip(self, flip: bool):
        self.flip = flip

    def is_open(self):
        return self.video.isOpened() and not self.stopped

    def draw(self, frame):
        if frame is None:
            return
        cv2.imshow(self.camera_name, frame)
        cv2.waitKey(1)

    def close(self):
        print(f"closing video camera {self.camera_name}")
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join() # Wait for background thread to finish
        if self.video:
            self.video.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

# --- Updated Main Usage ---

async def main(video_manager):
    print("Starting main loop...")
    
    while video_manager.is_open():
        # 1. READ PHASE: 
        # This is now nearly instant (~0.5ms vs 30ms) because it's just a memory copy.
        # We don't need 'await' or executors anymore because it doesn't block!
        frame = video_manager.capture_frame()
        
        if frame is None:
            await asyncio.sleep(0.01)
            continue
        
        # 2. RUN MODELS (Simulated):
        # Even if this takes 50ms, the background thread will keep updating 
        # 'latest_frame' so the next loop iteration gets fresh data.
        # results = model.process(frame) 

        # 3. DRAW
        video_manager.draw(frame)
        
        # Yield to event loop to keep things responsive
        await asyncio.sleep(0)

if __name__ == "__main__":
    # Ensure you have a "Camera_0" or change the string to match your device index
    try:
        with VideoManager("Camera_0", screen_xy=[0,0]) as vm:
            asyncio.run(main(vm))
    except KeyboardInterrupt:
        pass