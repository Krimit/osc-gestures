from aiohttp import web

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
        self.app.router.add_get('/style.css', self.handle_style) # New route for CSS
        self.app.router.add_get('/video_feed', self.handle_video_feed)
        self.app.router.add_get('/api/status', self.handle_status)

    async def handle_index(self, request):
        # Serve the HTML file directly from disk
        return web.FileResponse(os.path.join(self.base_dir, 'index.html'))

    async def handle_style(self, request):
        # Serve the CSS file directly from disk
        return web.FileResponse(os.path.join(self.base_dir, 'style.css'))

    async def handle_status(self, request):
        """API Endpoint that returns the current StreamState as JSON"""
        if self.stream_state:
            data = {
                "event": self.stream_state.event_number,
                "current": self.stream_state.current_gesture,
                "next": self.stream_state.next_gesture
            }
        else:
            data = {"event": -1, "current": "Error", "next": "No State"}
            
        return web.json_response(data)

    async def handle_video_feed(self, request):
        """MJPEG Streaming Endpoint"""
        boundary = "frame"
        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': f'multipart/x-mixed-replace; boundary={boundary}'
        })
        await response.prepare(request)

        last_sent_id = -1 

        try:
            while self._is_running:
                # Check if we have a new frame compared to last time
                if self.stream_state and self.stream_state.frame_bytes and self.stream_state.frame_id > last_sent_id:
                    frame_data = self.stream_state.frame_bytes
                    last_sent_id = self.stream_state.frame_id

                    await response.write(
                        f'--{boundary}\r\n'.encode() +
                        b'Content-Type: image/jpeg\r\n' +
                        f'Content-Length: {len(frame_data)}\r\n\r\n'.encode() +
                        frame_data + 
                        b'\r\n'
                    )
                    # aggressive wait: we can check frequently because check is cheap
                    await asyncio.sleep(0.016) 
                else:
                    # No new frame yet, sleep a bit to yield control
                    await asyncio.sleep(0.01)
                    
        except (ConnectionResetError, web.HTTPException):
            # Normal client disconnection
            print("[WEB] Client disconnected from video feed.")
            pass
        except Exception as e:
            # 4. Catch the crash! This usually reveals the 'NameError' or logic bug.
            print(f"[ERROR] Video Feed Crashed: {e}")
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