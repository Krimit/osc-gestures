import asyncio
from functools import wraps

def timeit_async(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get the current running event loop
        loop = asyncio.get_running_loop() 
        
        start_time = loop.time()  # Start time relative to the loop
        try:
            return await func(*args, **kwargs)
        finally:
            end_time = loop.time()
            total_ms = (end_time - start_time) * 1000
            print(f"Function {func.__name__} took {total_ms:.2f} ms")
    return wrapper
