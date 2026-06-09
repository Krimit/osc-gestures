from aiohttp import web

import time
import asyncio
import json
import os
import weakref

import logging
logging.getLogger('aiohttp.access').setLevel(logging.WARNING)

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

    async def on_shutdown(self, app):
        """Explicitly close all active WebSockets so cleanup() doesn't hang."""
        for ws in set(app['websockets']):
            await ws.close(code=1001, message='Server shutting down')


    def setup_routes(self):
        self.app['websockets'] = weakref.WeakSet()
        # Register the shutdown hook
        self.app.on_shutdown.append(self.on_shutdown)
        
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/style.css', self.handle_style)
        self.app.router.add_get('/performance_instruction', self.handle_websocket)

        # Dynamic route: /video/0, /video/1, etc.
        self.app.router.add_get('/video/{id}', self.handle_video_feed)

    async def handle_index(self, request):
        # Serve the HTML file directly from disk
        return web.FileResponse(os.path.join(self.base_dir, 'index.html'))

    async def handle_style(self, request):
        # Serve the CSS file directly from disk
        return web.FileResponse(os.path.join(self.base_dir, 'style.css'))


    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Track it!
        self.app['websockets'].add(ws)

        # Background task to push updates to the client
        async def send_updates():
            try:
                while not ws.closed:
                    if not self.stream_state:
                        data = {"event": -1, "current": "Error", "next": "No State", "fps_gpu": 0, "mps_osc": 0, "active_videos": []}
                    else:
                        active_ids = list(self.stream_state.frame_bytes.keys())
                        data = {
                            "event": self.stream_state.event_number,
                            "current": self.stream_state.current_gesture,
                            "next": self.stream_state.next_gesture,
                            "fps_gpu": self.stream_state.fps_gpu,
                            "mps_osc": self.stream_state.mps_osc,
                            "active_videos": active_ids,
                        }
                    await ws.send_json(data)
                    await asyncio.sleep(0.1)
            except Exception:
                pass # Client disconnected or network dropped

        update_task = asyncio.create_task(send_updates())

        try:
            async for msg in ws:
                pass # Just keep the connection open
        finally:
            update_task.cancel()

        return ws


    async def handle_video_feed(self, request):
        video_id = request.match_info.get('id')
        
        # Set up a streaming response
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                'Cache-Control': 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0',
                'Pragma': 'no-cache'
            }
        )
        await response.prepare(request)

        try:
            while self._is_running:
                frame_data = self.stream_state.frame_bytes.get(video_id)
                if frame_data:
                    # Format the JPEG as a multipart frame
                    part_boundary = (
                        f"--frame\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(frame_data)}\r\n\r\n"
                    ).encode('utf-8')
                    
                    try:
                        async with asyncio.timeout(0.05): # 50ms is usually safer for network writes than 5ms
                            await response.write(part_boundary + frame_data + b'\r\n')
                    except asyncio.TimeoutError:
                        # The TCP buffer is full (backpressure). 
                        # We can log it, sleep briefly, and try the next frame on the next loop.
                        await asyncio.sleep(0.01)
                        continue 

                # Match your stream's framerate (e.g., 30fps)
                await asyncio.sleep(1/30) 
                
        except (ConnectionResetError, asyncio.CancelledError):
            pass # Client disconnected

        return response


    async def start(self):
        self.runner = web.AppRunner(self.app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        print(f"--- WEB INTERFACE READY at http://0.0.0.0:{self.port} ---")
        await site.start()

    async def stop(self):
        self._is_running = False
        if self.runner:
            await self.runner.cleanup()