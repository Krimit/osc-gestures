from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_message_builder import OscMessageBuilder

import cv2
from typing import List, Any
import asyncio
import numpy as np
import time
from setup_cameras import CameraSetup
from model_controller import ModelController, Detector
import sys
import concurrent.futures

from method_timer import timeit_async

from pythonosc.udp_client import SimpleUDPClient

from metal_video_bridge import MetalVideoBridge
from syphon import SyphonMetalServer
import objc

from typing import NamedTuple


# set this to true to debug the raw frame (which is sent to Metal) 
INCLUDE_ORIGINAL_FRAME_IN_GUI = False

COMPUTE_SEGMENT = True

 # Shared executor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

ip = "127.0.0.1"
recieve_port = 5061
send_port = 5056
send_client = SimpleUDPClient(ip, send_port)

class ModelKey(NamedTuple):
    camera_name: str
    detector: Detector

    def __str__(self):
        return f"{self.camera_name}_{self.detector.name}"

model_controllers: dict[ModelKey, ModelController] = {}
running_tasks: dict[ModelKey, asyncio.Task] = {}
_active_tasks = set()

def get_model_key(camera_name, detector):
    return f"{camera_name}_{detector.name}"

detectors_to_enabled = {}

latest_detections = {}

syphon_bridges = {}

in_setup_phase = False

camera_setup = CameraSetup()

W, H = 1280, 720
SEND_TO_TD = True

class NetworkStats:
    def __init__(self):
        self.start_time = time.time()
        self.msg_out_count = 0
        self.msg_out_bytes = 0
        self.metal_out_count = 0
        
        # Snapshot values (for display)
        self.display_out_rate = 0
        self.display_out_kbps = 0.0
        self.metal_out_rate = 0

    def check_tick(self):
        """Updates averages if 1 second has passed"""
        now = time.time()
        elapsed = now - self.start_time
        if elapsed >= 1.0:
            self.display_out_rate = self.msg_out_count / elapsed
            self.display_out_kbps = (self.msg_out_bytes * 8) / 1000 / elapsed # Kbps
            self.metal_out_rate = self.metal_out_count / elapsed
            
            # Reset
            self.msg_out_count = 0
            self.msg_out_bytes = 0
            self.metal_out_count = 0
            self.start_time = now

    def record_metal(self):
        """Call this right before publishing to GPU metal"""
        self.metal_out_count += 1

    def record_send(self, address, args):
        """Call this right before client.send_message"""
        self.msg_out_count += 1
        # Estimate size: Address string + 4 bytes per float/int arg + 20 bytes overhead
        size_est = len(address) + 20
        if isinstance(args, list):
            size_est += len(args) * 4
        else:
            size_est += 4
        self.msg_out_bytes += size_est  

    def record_no_send(self):
        """Call this when we have a loop iteration with nothing sent"""
        return      


# Initialize globally
net_stats = NetworkStats()



def handle_cameras(address: str, *args: List[Any]) -> None:
    print("address: {}, message: {}".format(address, args))
    # We expect an even number of args. For "select" should be pairs of camera_name, model_type, no args for the other commands.
    if not len(args) % 2 == 0:
        print("Unexpected dispatcher arguments for {}: {}".format(address, args))
        return

    # Check that address is expected
    if not address in ["/controller/cameras/start", "/controller/cameras/stop", "/controller/cameras/select"]:
        print("Unexpected dispatcher address: {}".format(address))
        return

    global cameras_enabled
    global camera_name_to_detector
    global in_setup_phase
    command = address.removeprefix("/controller/cameras/")
    if command == "start":
        print("Starting the cameras.")
        camera_setup.start_all_videos()
        in_setup_phase = True
    elif command == "stop":
        print("Stopping the Cameras.")
        camera_setup.close()
        in_setup_phase = False
    else:
        print("unrecognized command: " + command)
  

def setup_selected_models(camera_detector_pairs: list, camera_name_to_camera: dict) -> None:
    global model_controllers
    # Clear the specific category if needed, or just update
    print(f"setup_selected_models pairs {camera_detector_pairs} {camera_name_to_camera}")
    for camera_name, detector in camera_detector_pairs:
        print("initializing models {} {}".format(camera_name, detector))
        if camera_name != "None":
            # Add to flat registry
            model_controllers[(camera_name, detector)] = ModelController(
                camera_name_to_camera[camera_name], detector, executor
            )
        print("initialized model controller for detector {}".format(detector))
    print("finished initializing model controllers for detectors {}".format(model_controllers.keys()))    


def setup_segment_models(camera_name_to_camera: dict) -> None:
    global model_controllers
    print(f"setup_segment_models {camera_name_to_camera}")
    for cam_name, camera in camera_name_to_camera.items():
        if cam_name != "None":
            # Registry Key: ('Camera_0', Detector.SEGMENT)
            key = (cam_name, Detector.SEGMENT)
            if key not in model_controllers:
                model_controllers[key] = ModelController(camera, Detector.SEGMENT, executor)

def handle_models(address: str, *args: List[Any]) -> None:
    print("address: {}, message: {}".format(address, args))

    expected_addresses = ["/controller/models/assign", 
    "/controller/models/HANDS/on", "/controller/models/HANDS/off", 
    "/controller/models/FACE/on", "/controller/models/FACE/off", 
    "/controller/models/HANDS_AND_FACE/on", "/controller/models/HANDS_AND_FACE/off"]


    if not address in expected_addresses:
        print("Unexpected dispatcher address: {}".format(address))
        return

    global detectors_to_enabled
    model_instruction = address.removeprefix("/controller/models/").split("/")

    if model_instruction[0] == "assign":
        print("Selecting models to use and pairing with cameras.")
        if not len(args) >= 2 and not len(args) % 2 == 0:
            print("Unexpected args, must have at least one camera to use, and must have camera-model pairs.")
            return
        camera_detector_pairs = []
        for i in range(0, len(args), 2):
            camera_name = args[i]
            detector_type = Detector[args[i+1]]
            if camera_name != "None":
                camera_detector_pairs.append((camera_name, detector_type))

        print("will use cameras to detectors: {}".format(camera_detector_pairs))
        detectors_to_enabled = {pair[1] : False for pair in camera_detector_pairs}
        print("Initial detector states: {}".format(detectors_to_enabled))
        unique_cameras = set(pair[0] for pair in camera_detector_pairs)
        camera_setup.stop_unused_cameras(list(unique_cameras))
        
        #This needs to be revisisted, multiple models might need different orientations, so this should happen in the model instead
        #camera_setup.set_camera_orientation_by_model(camera_name_to_detector)
        setup_selected_models(camera_detector_pairs, camera_setup.video_managers)
        setup_segment_models(camera_setup.video_managers)
        print("Done assigning models.")
        return


    command = model_instruction[1]
    detector = Detector[model_instruction[0]]
    if command == "on":
        print("Starting the {} model.".format(detector))
        detectors_to_enabled[detector] = True
    elif command == "off":
        print("Stopping the Hands model.") 
        detectors_to_enabled[detector] = False
    else:
        print("unrecognized command: " + command)            

dispatcher = Dispatcher()

# @deprecated
# Test the cameras, to assign cameras to models.
#dispatcher.map("/controller/camera-setup*", deprecated_handle_camera)

# Start the cameras, to assign cameras to models.
dispatcher.map("/controller/cameras*", handle_cameras)

# handle the model lifecycle to start or stop them
dispatcher.map("/controller/models*", handle_models)

def compute_60fps_sleep_time(start_time):
    target_fps = 60
    ideal_frame_duration = 1.0 / target_fps  # 0.01666...
    end_time = asyncio.get_event_loop().time()
    loop_duration = end_time - start_time
    sleep_time = ideal_frame_duration - loop_duration
    if sleep_time > 0:
        # We finished early! Sleep only the remainder.
        return sleep_time
    else:
        # We are running slow! Don't sleep at all, yield to others.
        return 0.001

async def model_supervisor():
    """Manages the lifecycle of model workers using ModelKey abstraction."""
    while True:
        # 1. Identify what should be running based on current registry
        desired_keys = set(model_controllers.keys())
        active_keys = set(running_tasks.keys())

        # 2. Spawn new models
        for key in desired_keys - active_keys:
            controller = model_controllers[key]
            print(f"[SUPERVISOR] Spawning worker: {key}")
            
            task = asyncio.create_task(model_worker(controller))
            running_tasks[key] = task
            _active_tasks.add(task)
            task.add_done_callback(_active_tasks.discard)

        # 3. Kill removed models
        for key in active_keys - desired_keys:
            print(f"[SUPERVISOR] Pruning worker: {key}")
            running_tasks[key].cancel()
            del running_tasks[key]

        await asyncio.sleep(0.5)


def is_detector_active(camera_name, detector_type):
    """
    Logic to determine if a specific controller should be processing frames.
    """
    # 1. If it's a primary model (FACE/HANDS), check the global toggle
    if detector_type != Detector.SEGMENT:
        return detectors_to_enabled.get(detector_type, False)

    # 2. If it's a SEGMENTER, check if it's manually on OR if 
    # ANY primary model on this specific camera is on.
    if detectors_to_enabled.get(Detector.SEGMENT, False):
        return True
        
    # Check other models on this same camera
    for (cam, det) in model_controllers.keys():
        if cam == camera_name and det != Detector.SEGMENT:
            if detectors_to_enabled.get(det, False):
                return True
                
    return False


@timeit_async
async def detect(model_controller):
    global latest_detections
    """Detection iteration"""
    if model_controller.is_open() and detectors_to_enabled[model_controller.enabled_detector]:
        detection = await model_controller.detect()
        if detection is not None:
            entry = {model_controller.name : detection}
            latest_detections.update(entry)

            osc_messages = detection.detection_dict

            if osc_messages:
                for key, message in osc_messages.items():
                    if message is not None:
                        path = "/detect/" + key
                        print("sending osc message to {}".format(path))
                        net_stats.record_send(path, message)
                        send_client.send_message(path, message)
            else:
                net_stats.record_no_send()

async def model_worker(controller):
    """Isolated loop for a single model/camera pairing."""
    try:
        while True:
            # DYNAMIC CHECK: Is my camera-specific task needed right now?
            if not is_detector_active(controller.video_manager.camera_name, controller.enabled_detector):
                print(f"detector {controller.enabled_detector} is not active")
                await asyncio.sleep(0.1)
                continue
                
            start_time = asyncio.get_event_loop().time()
            detection = await controller.detect()
            
            if detection:
                entry = {controller.name : detection}
                latest_detections.update(entry)
                osc_messages = detection.detection_dict
                if osc_messages:
                    for key, message in osc_messages.items():
                        if message is not None:
                            path = "/detect/" + key
                            print("sending osc message to {}".format(path))
                            send_client.send_message(path, message)
                            net_stats.record_send(path, message)
                else:
                    net_stats.record_no_send()
            await asyncio.sleep(compute_60fps_sleep_time(start_time))
    except asyncio.CancelledError:
        controller.close()

async def camera_selection():
    while True:
        while camera_setup.is_open() and in_setup_phase:
            print("camera setup running")
            camera_setup.do_loop(True)
            await asyncio.sleep(0.001)   
        await asyncio.sleep(0.1)


def draw_hud(frame, stats):
    """
    Prepends a black header and writes stats.
    Returns the new frame with the header.
    """
    height, width = frame.shape[:2]
    header_height = 40
    
    # 1. Create a black background for the header
    # (We could simply draw a rectangle, but padding the image is safer for aspect ratio)
    header = cv2.copyMakeBorder(
        frame, 
        top=header_height, 
        bottom=0, 
        left=0, 
        right=0, 
        borderType=cv2.BORDER_CONSTANT, 
        value=(0, 0, 0) # Black
    )

    # Calculate stats
    net_stats.check_tick()
    
    # 2. Format the text
    text = f"OSC out {int(stats.display_out_rate)} mps | {stats.display_out_kbps:.1f}  Kbps | GPU out {int(stats.metal_out_rate)} fps"
    
    # 3. Dynamic color: Green if healthy (<600 msgs), Red if high load
    color = (0, 255, 0) if stats.display_out_rate < 600 else (0, 0, 255)
    
    # 4. Put Text
    cv2.putText(
        header, 
        text, 
        (10, 25), # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.4, # Font Scale
        color, 
        1, # Thickness
        cv2.LINE_AA
    )
    
    return header

def resize_frame(frame):
    # Optional: Resize to ensure they match widths
    target_w = 640
    h, w = frame.shape[:2]
    return cv2.resize(frame, (target_w, int(h * target_w / w)))

@timeit_async
async def gui_manager():
    global latest_detections
    window_name = "Hollow Man Debug View"
    cv2.namedWindow(window_name)

    while True:
        start_time = asyncio.get_event_loop().time()
        frames_to_stack = []
        
        # Pull frames from the shared dictionary
        # Sorting by key ensures the order (top to bottom) stays consistent
        for name in sorted(latest_detections.keys()):
            detection = latest_detections[name]
            frame = detection.annotated_frame
            if frame is not None:
                frame = resize_frame(frame)
                frames_to_stack.append(frame)

            if INCLUDE_ORIGINAL_FRAME_IN_GUI:
                if detection.original_frame is not None:
                    original_frame = resize_frame(detection.original_frame)
                    frames_to_stack.append(original_frame)

        if frames_to_stack:
            # Stack all frames vertically
            # Note: All frames must have the same width and channel count
            combined_view = np.vstack(frames_to_stack)
            final_image = draw_hud(combined_view, net_stats)
            cv2.imshow(window_name, final_image)

        # ONE waitKey call to rule them all
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        await asyncio.sleep(compute_60fps_sleep_time(start_time))

def publish_to_metal(bridge, frame):
    with objc.autorelease_pool():        
        bridge.publish_to_metal(frame)

@timeit_async
async def syphon_manager():
    global latest_detections
    global syphon_bridges

    loop = asyncio.get_running_loop()

    if SEND_TO_TD:
        while True:
            start_time = asyncio.get_event_loop().time()
            current_names = sorted(list(latest_detections.keys()))
            if len(current_names) > 0:
                net_stats.record_metal()
            for name in current_names:
                detection = latest_detections[name]
                if detection.name not in syphon_bridges:
                    # Initialize bridge with specific camera dimensions if needed
                    output_name = "HollowManVideo_" + detection.name
                    syphon_bridges[detection.name] = MetalVideoBridge(W, H, output_name)
                    print("Created new MetalVideoBridge, sending video to metal as: {}".format(output_name))

                frame = detection.original_frame
                if frame is None:
                    continue

                await loop.run_in_executor(executor, publish_to_metal, syphon_bridges[detection.name], frame)
            await asyncio.sleep(compute_60fps_sleep_time(start_time))


def inject_osc_message(address, args_list):
    """
    Wraps data into an OSC packet and feeds it to the dispatcher.
    This triggers all wildcard matches (e.g., /controller/cameras*).
    """
    builder = OscMessageBuilder(address=address)
    for val in args_list:
        builder.add_arg(val)
    
    msg = builder.build()
    # .dgram is the raw byte packet. 
    # We pass a dummy IP/Port as the "sender".
    dispatcher.call_handlers_for_packet(msg.dgram, ("127.0.0.1", 0))

async def test_sequence_injector(model_mapping):
    """
    Simulates a sequence of OSC messages to automate the setup process.
    """
    if not TEST_MODE:
        return

    print("\n[TEST LAYER] Starting automated test sequence...")
    await asyncio.sleep(1)

    # 1. Start the cameras
    print("[TEST LAYER] camera setup starting: /controller/cameras/start")
    inject_osc_message("/controller/cameras/start", [])
    await asyncio.sleep(0.5)
    print("[TEST LAYER] camera setup done: /controller/cameras/stop")
    inject_osc_message("/controller/cameras/stop", [])


    print(f"[TEST LAYER] Starting models: /controller/models/assign {model_mapping}")
    inject_osc_message("/controller/models/assign", model_mapping)
    await asyncio.sleep(2)

    # 3. Dynamically enable every model mentioned in the assignment
    # We step by 2, looking at index 1, 3, 5...
    for i in range(1, len(model_mapping), 2):
        model_name = model_mapping[i]
        osc_path = f"/controller/models/{model_name}/on"
        
        print(f"[TEST LAYER] Enabling model: {osc_path}")
        inject_osc_message(osc_path, [])
        
        # Small delay between activations to prevent race conditions
        await asyncio.sleep(0.5)
    
    print("[TEST LAYER] Sequence complete. Server is now running in test state.\n")

async def cleanup():
    """Gracefully shuts down all components in the correct order"""
    print("\n--- Starting Graceful Shutdown ---")
    
    # 1. Stop the cameras first (stops the data source)
    print("Closing cameras...")
    camera_setup.close()

    # 2. Close Metal Bridges (Crucial: Stop Syphon before Metal device is killed)
    print(f"Closing {len(syphon_bridges)} Syphon bridges...")
    for name, bridge in syphon_bridges.items():
        try:
            bridge.close() # This calls syphon_server.stop()
        except Exception as e:
            print(f"Error closing bridge {name}: {e}")
    syphon_bridges.clear()

    # 3. Close Model Controllers
    print(f"Closing {len(model_controllers)} model controllers...")
    for (cam_name, detector), controller in model_controllers.items():
        try:
            controller.close()
        except Exception as e:
            print(f"Error closing controller {detector}: {e}")
    model_controllers.clear()

    # 4. Shutdown the ThreadPoolExecutor
    print("Shutting down executor...")
    executor.shutdown(wait=True, cancel_futures=True)
    
    # 5. Destroy OpenCV Windows
    cv2.destroyAllWindows()
    print("--- Shutdown Complete ---")

# When enabled, test python code without a MaxMsp dependancy. Turn this OFF when using MaxMsp!
TEST_MODE = True

async def main():    
    server = AsyncIOOSCUDPServer((ip, recieve_port), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    tasks = [
        model_supervisor(),       # Manages model worker lifecycles
        camera_selection(), # Manages camera setup phase
        gui_manager(),      # OpenCV debug window
        syphon_manager(),   # GPU / Syphon output
    ]

    # Adjust this as needed when testing
    #model_mapping = ["Camera_0", "FACE"]
    #model_mapping = ["Camera_0", "HANDS"]
    #model_mapping = ["Camera_0", "FACE", "Camera_1", "HANDS"]
    model_mapping = ["Camera_0", "FACE", "Camera_0", "HANDS"]


    if TEST_MODE:
        tasks.append(test_sequence_injector(model_mapping))

    try:
        # Using return_exceptions=True can prevent one task crash from killing the whole app, but we don't want that generally!
        await asyncio.gather(*tasks, return_exceptions=False)
    except asyncio.CancelledError:
        pass
    finally:
        transport.close()
        await cleanup()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This catch prevents the ugly traceback on Ctrl+C
        pass