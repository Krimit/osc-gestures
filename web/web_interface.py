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
        
        # Just get the current frame data once
        frame_data = self.stream_state.frame_bytes.get(video_id)
        
        if not frame_data:
            return web.Response(status=404)

        # Return as a standard single JPEG image
        return web.Response(
            body=frame_data,
            content_type='image/jpeg',
            headers={
                'Cache-Control': 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0',
                'Pragma': 'no-cache'
            }
        )


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