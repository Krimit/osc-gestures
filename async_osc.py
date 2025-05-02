from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from typing import List, Any
import asyncio
import ghands_streaming
import setup_cameras
from model_controller import ModelController, Detector
#from model_controller import Detector

from pythonosc.udp_client import SimpleUDPClient


ip = "127.0.0.1"
recieve_port = 5060
send_port = 5056
send_client = SimpleUDPClient(ip, send_port)

is_hands_enabled = False

hands_model_controller = None

is_setup_enabled = False

def handle_hands(address: str, *args: List[Any]) -> None:
    print("address: {}, message: {}".format(address, args))
    # We expect one string argument
    if not len(args) in [0, 1]:
        print("Unexpected dispatcher arguments for {}: {}".format(address, args))
        return

    if not address in ["/controller/hands/start", "/controller/hands/stop"]:
        print("Unexpected dispatcher address: {}".format(address))
        return

    global is_hands_enabled
    command = address.removeprefix("/controller/hands/")
    if command == "start":
        print("Enabling the Hands model.")
        is_hands_enabled = True
    elif command == "stop":
        print("Disabling the Hands model.") 
        is_hands_enabled = False
    else:
        print("unrecognized command: " + command)    


def handle_camera(address: str, *args: List[Any]) -> None:
    print("address: {}, message: {}".format(address, args))
    # We expect one string argument
    if not len(args) == 0:
        print("Unexpected dispatcher arguments for {}: {}".format(address, args))
        return

    # Check that address starts with filter
    if not address in ["/controller/camera-setup/off", "/controller/camera-setup/on"]:
        print("Unexpected dispatcher address: {}".format(address))
        return

    global is_setup_enabled
    command = address.removeprefix("/controller/camera-setup/")
    if command == "on":
        print("Starting the Camera Setup.")
        is_setup_enabled = True
    elif command == "off":
        print("Ending the Camera Setup.")
        is_setup_enabled = False
    else:
        print("unrecognized command: " + command)

def handle_models(address: str, *args: List[Any]) -> None:
    print("address: {}, message: {}".format(address, args))
    # We expect one string argument
    if not len(args) in [0, 1]:
        print("Unexpected dispatcher arguments for {}: {}".format(address, args))
        return

    if not address in ["/controller/models/hands/on", "/controller/models/hands/off"]:
        print("Unexpected dispatcher address: {}".format(address))
        return

    global is_hands_enabled
    global hands_model_controller
    model_instruction = address.removeprefix("/controller/models/").split("/")
    if len(model_instruction) != 2:
        print("Unexpected model instruction address for {}: {}".format(address, args))
    command = model_instruction[1]
    if model_instruction[0] == "hands":
        if command == "on":
            if not len(args) == 1:
                print("Unexpected args, missing camera name")
            print("Starting the Hands model.")
            hands_camera_name = args[0]
            hands_model_controller = ModelController(hands_camera_name, Detector.HANDS)
        elif command == "off":
            print("Stopping the Hands model.") 
            if hands_model_controller is not None:
                hands_model_controller.close()
            is_hands_enabled = False
            hands_model_controller = None
        else:
            print("unrecognized command: " + command)            

dispatcher = Dispatcher()
# Hand gesture recognition
dispatcher.map("/controller/hands*", handle_hands)

# Test the cameras, to assign cameras to models.
dispatcher.map("/controller/camera-setup*", handle_camera)

# handle the model lifecycle to start or stop them
dispatcher.map("/controller/models*", handle_models)



def detect(model_controller):
    """Detection iteration"""
    if model_controller.is_open() and is_hands_enabled:
        print("akrim are we looping?")
        osc_messages = model_controller.detect()
        for message in osc_messages:
            if message is not None:
                send_client.send_message("/detect", message)       


async def hands():
    """Main program loop"""
    while True:
        while hands_model_controller is not None and hands_model_controller.is_open() and is_hands_enabled:
            print("akrim are we looping?")
            detect(hands_model_controller)  
            await asyncio.sleep(0)
        await asyncio.sleep(0)            


async def camera_setup():
    """Main program loop"""
    global is_setup_enabled
    while True:
        if is_setup_enabled:
            with setup_cameras.CameraSetup() as setup:
                setup.start_all_videos()
                #is_setup_enabled = False
                while setup.is_open() and is_setup_enabled:
                    setup.do_loop(True)
                    await asyncio.sleep(0)   
            print("Finished camera setup")             
        await asyncio.sleep(0)



async def main():
    server = AsyncIOOSCUDPServer((ip, recieve_port), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    await asyncio.gather(camera_setup(), hands())
    # loop = asyncio.get_event_loop()
    # asyncio.set_event_loop(loop)
    # loop.create_task(camera_setup())
    # #loop.create_task(hands())
    # await loop.run_forever()

    # await loop_setup()  # Enter main loop of program

    transport.close()  # Clean up serve endpoint


if __name__ == '__main__':
    asyncio.run(main())