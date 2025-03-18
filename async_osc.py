from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from typing import List, Any
import asyncio
import ghands_streaming
import setup_cameras


ip = "127.0.0.1"
port = 5060

is_hands_enabled = False
is_setup_enabled = False

def handle_controller(address: str, *args: List[Any]) -> None:
    print("address: {}, message: {}".format(address, args))
    # We expect one string argument
    if not len(args) == 1 or type(args[0]) is not str:
        print("Unexpected dispatcher arguments: {}".format(args))
        return

    # Check that address starts with filter
    if not address == "/controller":
        print("Unexpected dispatcher address: {}".format(address))
        return

    global is_hands_enabled
    global is_setup_enabled
    command = args[0]
    if command == "hands-on":
        print("will enable the Hands model.")
        is_hands_enabled = True
    if command == "hands-off":
        print("will disable the Hands model.") 
        is_hands_enabled = False
    if command == "start-setup":
        is_setup_enabled = True


dispatcher = Dispatcher()
dispatcher.map("/controller*", handle_controller)


async def loop():
    """Main program loop"""
    with ghands_streaming.Mediapipe_HandsModule() as hands_module:
        #hands_module.init()
        while hands_module.is_open():
            hands_module.do_loop(is_hands_enabled)      
            await asyncio.sleep(0)


async def loop_setup():
    """Main program loop"""
    global is_setup_enabled
    while True:
        if is_setup_enabled:
            with setup_cameras.CameraSetup() as setup:
                setup.start_all_videos()
                is_setup_enabled = False
                while setup.is_open():
                    setup.do_loop(True)
                    await asyncio.sleep(0)   
            print("out of setup loop")             
        await asyncio.sleep(0)



async def init_main():
    server = AsyncIOOSCUDPServer((ip, port), dispatcher, asyncio.get_event_loop())
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving

    await loop_setup()  # Enter main loop of program

    transport.close()  # Clean up serve endpoint


asyncio.run(init_main())

