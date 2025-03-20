from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from typing import List, Any
import asyncio
import ghands_streaming
import setup_cameras


ip = "127.0.0.1"
port = 5060

is_hands_enabled = False
hands_camera_name = ""

is_setup_enabled = False

def handle_hands(address: str, *args: List[Any]) -> None:
    print("address: {}, message: {}".format(address, args))
    # We expect one string argument
    if not len(args) in [0, 1]:
        print("Unexpected dispatcher arguments for {}: {}".format(address, args))
        return

    if not address in ["/controller/hands/on", "/controller/hands/off"]:
        print("Unexpected dispatcher address: {}".format(address))
        return

    global is_hands_enabled
    global hands_camera_name
    command = address.removeprefix("/controller/hands/")
    if command == "on":
        if not len(args) == 1:
            print("Unexpected args, missing camera name")
        print("Enabling the Hands model.")
        hands_camera_name = args[0]
        is_hands_enabled = True
    elif command == "off":
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

dispatcher = Dispatcher()
# Hand gesture recognition
dispatcher.map("/controller/hands*", handle_hands)

# Test the cameras, to assign cameras to models.
dispatcher.map("/controller/camera-setup*", handle_camera)



async def hands():
    """Main program loop"""
    while True:
        if is_hands_enabled:
            with ghands_streaming.Mediapipe_HandsModule(hands_camera_name) as hands_module:
                while hands_module.is_open() and is_hands_enabled:
                    hands_module.do_loop(is_hands_enabled)      
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
    server = AsyncIOOSCUDPServer((ip, port), dispatcher, asyncio.get_event_loop())
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