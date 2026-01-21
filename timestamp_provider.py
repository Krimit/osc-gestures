import time
import threading

class TimestampProvider:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Thread-safe Singleton pattern
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TimestampProvider, cls).__new__(cls)
                # Initialize variables here since __init__ runs every call
                cls._instance.last_timestamp = 0
            return cls._instance

    def get_timestamp(self):
        with self._lock:
            # MediaPipe expects microseconds or milliseconds (integers)
            current_time = int(time.time() * 1000)

            # Ensure the new timestamp is strictly greater than the last
            if current_time <= self.last_timestamp:
                self.last_timestamp += 1
            else:
                self.last_timestamp = current_time

            return self.last_timestamp