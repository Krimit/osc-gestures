from aiohttp import web

import time
import asyncio
import json
import os

class WebInterface:
    def __init__(self, port=8191, stream_state=None):
        self.port = port
        self.stream_state = stream_state
        self.app = web.Application()
        self.runner = None
        self._is_running = True
        
        # Determine the directory this script lives in to safely find the files
        self.base_dir = os.path.dirname(__file__)
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/style.css', self.handle_style)
        self.app.router.add_get('/api/status', self.handle_status)
        # Dynamic route: /video/0, /video/1, etc.
        self.app.router.add_get('/video/{id}', self.handle_video_feed)

    async def handle_index(self, request):
        # Serve the HTML file directly from disk
        return web.FileResponse(os.path.join(self.base_dir, 'index.html'))

    async def handle_style(self, request):
        # Serve the CSS file directly from disk
        return web.FileResponse(os.path.join(self.base_dir, 'style.css'))


    async def handle_status(self, request):
        if not self.stream_state:
            data = {"event": -1, "current": "Error", "next": "No State", "fps_gpu": 0, "mps_osc": 0, "active_videos": []}
        else:
            # Send the actual controller names that are active
            active_ids = list(self.stream_state.frame_bytes.keys())
            data = {
                "event": self.stream_state.event_number,
                "current": self.stream_state.current_gesture,
                "next": self.stream_state.next_gesture,
                "fps_gpu": self.stream_state.fps_gpu,
                "mps_osc": self.stream_state.mps_osc,
                "active_videos": active_ids,
            }
        return web.json_response(data)


    async def handle_video_feed(self, request):
        video_id = request.match_info.get('id')
        boundary = "frame"
        response = web.StreamResponse(status=200, headers={
            'Content-Type': f'multipart/x-mixed-replace; boundary={boundary}'
        })
        await response.prepare(request)

        last_sent_id = -1

        last_sent_time = 0
        TARGET_WEB_FPS = 24
        frame_interval = 1.0 / TARGET_WEB_FPS
        
        try:
            while self._is_running:
                # Get the frame specific to this ID
                frame_data = self.stream_state.frame_bytes.get(video_id)
                current_id = self.stream_state.frame_ids.get(video_id, -1)
                current_time = time.time()

                # Only send if it's a new frame AND enough time has passed
                if frame_data and current_id != last_sent_id and (current_time - last_sent_time) >= frame_interval:
                    last_sent_id = current_id
                    last_sent_time = current_time
                    start_push = time.time()

                    await response.write(
                        f'--{boundary}\r\nContent-Type: image/jpeg\r\n'
                        f'Content-Length: {len(frame_data)}\r\n\r\n'.encode() +
                        frame_data + b'\r\n'
                    )
                    # If the router is jammed, this forces Python to let go and drop the frame.
                    try:
                        await asyncio.wait_for(response.drain(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass # Network is congested, silently drop this frame and keep the loop moving
                    push_duration = time.time() - start_push
                    if push_duration > 0.05: # Longer than 50ms
                        print(f"[NETWORK WARNING] Stream {video_id} took {push_duration:.3f}s to drain! iPad may be falling behind.")

                    await asyncio.sleep(0.001)
                else:
                    # No new frame yet, sleep a bit to yield control
                    # Can be completely normal, only log if debugging something.
                    #print(f"[WEB DEBUG] Stream {video_id} is starving/idle. current_id: {current_id}")
                    await asyncio.sleep(0.01)
        except (ConnectionResetError, web.HTTPException):
            # Normal client disconnection
            print("[WEB] Client disconnected from video feed.")
            pass
        except Exception as e:
            # 4. Catch the crash! This usually reveals the 'NameError' or logic bug.
            print(f"[ERROR] Video Feed Crashed: {e}")
            raise
        return response


    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        print(f"--- WEB INTERFACE READY at http://0.0.0.0:{self.port} ---")
        await site.start()

    async def stop(self):
        self._is_running = False
        if self.runner:
            await self.runner.cleanup()