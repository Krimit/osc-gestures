import cv2
import numpy as np
import Metal
import sys

class MetalVideoBridge:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.texture = None  # <--- We will store the texture here
        
        # 1. Get the GPU
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            print("Error: Metal is not supported.")
            sys.exit(0)
            
        # 2. Configure Descriptor
        self.texture_descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatBGRA8Unorm, 
            width, 
            height, 
            False
        )
        self.texture_descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
        self.texture = self.device.newTextureWithDescriptor_(self.texture_descriptor)

    def numpy_to_metal(self, frame):
        # 1. Prepare Data (Convert to BGRA)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # 2. Create the Texture ONLY ONCE
        # if self.texture is None:
        #     self.texture = self.device.newTextureWithDescriptor_(self.texture_descriptor)
        
        # 3. Update the existing texture's memory
        # This copies new pixels into the OLD texture object. Zero allocation.
        region = Metal.MTLRegionMake2D(0, 0, self.width, self.height)
        bytes_per_row = self.width * 4
        
        # Note: Using .tobytes() creates a CPU copy, but it's safe. 
        # For max performance, you'd use memoryview, but let's fix the leak first.
        self.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            frame.tobytes(),
            bytes_per_row
        )
        
        return self.texture