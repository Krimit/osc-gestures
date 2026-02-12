from aiohttp import web
import json

class WebInterface:
    def __init__(self, port=8080):
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/video_feed', self.handle_video_feed)
        self.app.router.add_post('/api/command', self.handle_command)

    def get_html(self):
        """
        Returns the HTML for the iPad Control Panel.
        Includes a responsive video stream and large touch-friendly buttons.
        """
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <style>
                body { background-color: #121212; color: #fff; font-family: sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
                #video-container { flex: 1; display: flex; justify-content: center; align-items: center; background: #000; overflow: hidden; }
                img { max-width: 100%; max-height: 100%; object-fit: contain; }
                
                #controls { height: 40%; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 15px; background: #1e1e1e; }
                
                .btn { 
                    border: none; border-radius: 12px; 
                    font-size: 18px; font-weight: bold; color: white;
                    cursor: pointer; transition: opacity 0.2s;
                    display: flex; align-items: center; justify-content: center;
                    text-transform: uppercase; letter-spacing: 1px;
                }
                .btn:active { opacity: 0.6; transform: scale(0.98); }
                
                /* Button Colors */
                .btn-green { background: #2e7d32; }
                .btn-red { background: #c62828; }
                .btn-blue { background: #1565c0; }
                .btn-grey { background: #424242; }

                /* Specific Grid Areas if needed */
                .full-width { grid-column: span 2; }
            </style>
        </head>
        <body>
            <div id="video-container">
                <img src="/video_feed" alt="Live Stream" />
            </div>
            
            <div id="controls">
                <button class="btn btn-green" onclick="sendCommand('/controller/models/HANDS/on')">Hands ON</button>
                <button class="btn btn-red"   onclick="sendCommand('/controller/models/HANDS/off')">Hands OFF</button>
                
                <button class="btn btn-green" onclick="sendCommand('/controller/models/FACE/on')">Face ON</button>
                <button class="btn btn-red"   onclick="sendCommand('/controller/models/FACE/off')">Face OFF</button>
                
                <button class="btn btn-blue full-width" onclick="sendCommand('/controller/cameras/start')">Start Cameras</button>
                <button class="btn btn-grey full-width" onclick="sendCommand('/controller/cameras/stop')">Stop Cameras</button>
            </div>

            <script>
                function sendCommand(addr, args=[]) {
                    fetch('/api/command', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({address: addr, args: args})
                    }).catch(console.error);
                }
            </script>
        </body>
        </html>
        """

    async def handle_index(self, request):
        return web.Response(text=self.get_html(), content_type='text/html')

    async def handle_command(self, request):
        """Receives JSON commands from iPad and injects them into the OSC system."""
        try:
            data = await request.json()
            address = data.get('address')
            args = data.get('args', [])
            
            print(f"[WEB] Received command: {address} {args}")
            
            # Reuse your existing injection logic!
            inject_osc_message(address, args)
            
            return web.Response(text="OK")
        except Exception as e:
            return web.Response(text=str(e), status=500)

    async def handle_video_feed(self, request):
        """MJPEG Streaming Endpoint"""
        boundary = "frame"
        response = web.StreamResponse(status=200, reason='OK', headers={
            'Content-Type': f'multipart/x-mixed-replace;boundary={boundary}'
        })
        await response.prepare(request)

        try:
            while True:
                # Non-blocking check for new frame
                async with stream_state.lock:
                    frame_data = stream_state.frame_bytes
                
                if frame_data:
                    await response.write(
                        f'--{boundary}\r\n'.encode() +
                        b'Content-Type: image/jpeg\r\n' +
                        f'Content-Length: {len(frame_data)}\r\n\r\n'.encode() +
                        frame_data + 
                        b'\r\n'
                    )
                    # Cap at ~30fps to save bandwidth
                    await asyncio.sleep(0.033)
                else:
                    await asyncio.sleep(0.1)
        except ConnectionResetError:
            # Normal when user closes tab
            pass
        return response

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '0.0.0.0', self.port)
        print(f"--- WEB INTERFACE READY at http://0.0.0.0:{self.port} ---")
        await site.start()

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()