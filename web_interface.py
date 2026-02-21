from aiohttp import web

import asyncio
import json

class WebInterface:
    def __init__(self, port=8191, stream_state=None):
        self.port = port
        self.stream_state = stream_state
        self.app = web.Application()
        self.runner = None
        self._is_running = True
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/video_feed', self.handle_video_feed)
        self.app.router.add_get('/api/status', self.handle_status)

    def get_html(self):
        """
        Returns HTML with a Top/Bottom Layout:
        [      VIDEO (75vh)       ]
        [ INFO PANEL (Auto/Row)   ]
        """
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
            <style>
                body { 
                    background-color: #000; 
                    color: #fff; 
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    margin: 0; 
                    padding: 0; 
                    width: 100vw; 
                    overflow-x: hidden;
                    overflow-y: auto; /* Allows scrolling if text gets extremely long */
                }

                /* --- TOP SIDE: VIDEO --- */
                #video-container { 
                    height: 75vh; /* STRICTLY lock video to 75% of the screen height */
                    width: 100%;
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    background: #000; 
                    border-bottom: 1px solid #333;
                }
                
                img { 
                    max-width: 100%; 
                    max-height: 100%; 
                    object-fit: contain; 
                    display: block; 
                }

                /* --- BOTTOM SIDE: INFO PANEL --- */
                #info-panel {
                    min-height: 25vh; /* Takes the rest of the screen, but can grow if needed */
                    width: 100%;
                    display: flex;
                    flex-direction: row; /* Side-by-side layout for the wide bottom bar */
                    flex-wrap: wrap; /* Safely wrap to a new line on very narrow screens */
                    justify-content: space-around;
                    align-items: flex-start;
                    padding: 20px;
                    background-color: #111;
                    box-sizing: border-box;
                    text-align: center;
                }

                .info-block {
                    margin: 10px 20px;
                    flex: 1 1 25%; /* Give each block equal room to breathe */
                }

                .label {
                    font-size: 14px;
                    text-transform: uppercase;
                    color: #888;
                    letter-spacing: 1px;
                    margin-bottom: 5px;
                }

                .value {
                    font-size: 32px;
                    font-weight: 700;
                    color: #eee;
                    
                    /* FIXED WRAPPING RULES */
                    white-space: pre-wrap;     /* Allows text to wrap into new lines natively */
                    overflow-wrap: break-word; /* Prevents long unbreakable words from blowing out the box */
                }

                /* Highlight the 'Next' gesture to make it pop for the performer */
                .value.next {
                    color: #4caf50; /* Green */
                }
                
                .value.event {
                    font-size: 60px; /* Huge Event Number */
                    color: #2196f3; /* Blue */
                }

            </style>
        </head>
        <body>
            
            <div id="video-container">
                <img src="/video_feed" alt="Live Stream" />
            </div>
            
            <div id="info-panel">
                <div class="info-block">
                    <div class="label">Event Number</div>
                    <div class="value event" id="event-display">--</div>
                </div>

                <div class="info-block">
                    <div class="label">Current Gesture</div>
                    <div class="value" id="current-display">Waiting...</div>
                </div>

                <div class="info-block">
                    <div class="label">Next Gesture</div>
                    <div class="value next" id="next-display">Waiting...</div>
                </div>
            </div>

            <script>
                // Function to poll the server for the latest text data
                async function updateStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();

                        document.getElementById('event-display').innerText = data.event;
                        document.getElementById('current-display').innerText = data.current || "-";
                        document.getElementById('next-display').innerText = data.next || "-";
                    } catch (e) {
                        console.error("Status fetch failed", e);
                    }
                }

                // Update every 200ms (5 times a second)
                setInterval(updateStatus, 200);
            </script>
        </body>
        </html>
        """

    async def handle_index(self, request):
        return web.Response(text=self.get_html(), content_type='text/html')

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