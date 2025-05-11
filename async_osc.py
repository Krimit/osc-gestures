from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from typing import List, Any
import asyncio
import time
from setup_cameras import CameraSetup
from model_controller import ModelController, Detector
#from model_controller import Detector

from pythonosc.udp_client import SimpleUDPClient


ip = "127.0.0.1"
recieve_port = 5060
send_port = 5056
send_client = SimpleUDPClient(ip, send_port)

model_controllers = {}

detectors_to_enabled = {}

in_setup_phase = False

camera_setup = CameraSetup()
 

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
        in_setup_phase = False
    else:
        print("unrecognized command: " + command)

def setup_selected_models(camera_name_to_detector: dict, camera_name_to_camera: dict) -> None:
    global model_controllers
    for camera_name, detector in camera_name_to_detector.items():
        print("initializing models {} {}".format(camera_name, detector))
        if camera_name is not None and camera_name != "None":
            model_controllers[detector] = ModelController(camera_name_to_camera[camera_name], detector)
            print("initialized model controller for dector {}".format(detector))
    print("finished initializing model controllers for detectors {}".format(model_controllers.keys()))    


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
        camera_name_to_detector = {args[i]: Detector[args[i+1]] for i in range(0, len(args), 2)}
        print("got cameras to detectors: {}".format(camera_name_to_detector))
        # None is a special camera name to mean we aren't using this model.
        del camera_name_to_detector["None"]
        print("will use cameras to detectors: {}".format(camera_name_to_detector))
        detectors_to_enabled = {d : False for d in camera_name_to_detector.values()}
        print("Initial detector states: {}".format(detectors_to_enabled))
        camera_setup.stop_unused_cameras(camera_name_to_detector.keys())
        setup_selected_models(camera_name_to_detector, camera_setup.video_managers)
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

def detect(model_controller):
    """Detection iteration"""
    if model_controller.is_open() and detectors_to_enabled[model_controller.enabled_detector]:
        osc_messages = model_controller.detect()
        if osc_messages is not None:
            for key, message in osc_messages.items():
                if message is not None:
                    path = "/detect/" + key
                    print("sending osc message {}: {}".format(path, message))
                    send_client.send_message(path, message)    

async def model(detector: Detector):
    """Main program loop"""
    last_iteration = time.time()
    while True:
        while detector in model_controllers.keys() and model_controllers[detector].is_open() and detectors_to_enabled[detector]:
            current_time = time.time()
            #print("--- {} model time since last iteration: {} ms".format(detector, current_time * 1000 - last_iteration * 1000))
            detect(model_controllers[detector])
            last_iteration = time.time()
            await asyncio.sleep(0)
        await asyncio.sleep(0)  


async def camera_selection():
    """Main program loop"""
    while True:
        while camera_setup.is_open() and in_setup_phase:
            print("camera setup running")
            camera_setup.do_loop(True)
            await asyncio.sleep(0)   
        await asyncio.sleep(0)



async def main():
    server = AsyncIOOSCUDPServer((ip, recieve_port), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    await asyncio.gather(camera_selection(), model(Detector.HANDS), model(Detector.FACE), model(Detector.HANDS_AND_FACE))

    transport.close()  # Clean up serve endpoint


if __name__ == '__main__':
    asyncio.run(main())