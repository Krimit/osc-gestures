import cv2
import numpy as np
import Metal
import sys
from syphon import SyphonMetalServer

class MetalVideoBridge:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # 1. Get the GPU (Metal Device)
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            print("Error: Metal is not supported on this device.")
            sys.exit(0)
            
        # 2. Configure the Texture Descriptor
        # We use BGRA8Unorm because OpenCV uses BGR natively, so adding Alpha (BGRA) is fast
        self.texture_descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatBGRA8Unorm, 
            width, 
            height, 
            False
        )
        # Allow CPU to write to this texture
        self.texture_descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)

    def numpy_to_metal(self, frame):
        # Safety Check: Syphon/Metal requires 4 channels (BGRA)
        # If the frame is standard OpenCV BGR, convert it here
        if frame.shape[2] == 3:
            # OpenCV defaults to BGR. We convert to BGRA.
            frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        """Converts a numpy array (BGRA) to a generic Metal Texture"""
        # Create a new empty texture on the GPU
        texture = self.device.newTextureWithDescriptor_(self.texture_descriptor)
        
        # Calculate memory layout
        bytes_per_row = self.width * 4  # 4 bytes for BGRA
        
        # Create a region that covers the whole image
        region = Metal.MTLRegionMake2D(0, 0, self.width, self.height)
        
        # Copy the raw bytes from the numpy array into the Metal texture
        # We use memoryview/tobytes ensures we send contiguous C-style memory
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            frame_bgra.tobytes(),
            bytes_per_row
        )
        return texture