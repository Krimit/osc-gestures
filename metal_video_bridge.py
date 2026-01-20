import cv2
import numpy as np
import Metal
import sys

from syphon import SyphonMetalServer


class MetalVideoBridge:
    def __init__(self, width: int, height: int, output_name: str):
        self.width = width
        self.height = height
        self.texture = None  # <--- We will store the texture here
        
        # 1. Get the GPU
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if not self.device:
            print("Error: Metal is not supported.")
            sys.exit(0)

        self.output_name = output_name
        self.syphon_server = SyphonMetalServer(output_name, device=self.device)    
            
        # 2. Configure Descriptor
        self.texture_descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatBGRA8Unorm, 
            width, 
            height, 
            False
        )
        self.texture_descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
        self.texture = self.device.newTextureWithDescriptor_(self.texture_descriptor)

    def close(self):
        print("cloing MetalVideoBridge " + self.output_name)
        self.syphon_server.stop()
        self.device = None

    def __exit__(self, unused_exc_type, unused_exc_value, unused_traceback):
        self.close()    


    #Convert CPU Array -> GPU Texture
    def numpy_to_metal(self, frame):
        # 1. Get current dimensions
        h, w = frame.shape[:2]

        # 2. FORCE RESIZE: If the frame doesn't match the Metal Texture, 
        # Metal will read past the end of the buffer or tilt the image.
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            w, h = self.width, self.height # Update variables

        # 1. Prepare Data (Convert to BGRA)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

        # Ensure the array is stored in one solid block of memory
        # 'ascontiguousarray' does NOTHING if it's already contiguous (zero cost)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        
        # 3. Update the existing texture's memory
        # This copies new pixels into the OLD texture object. Zero allocation.
        region = Metal.MTLRegionMake2D(0, 0, self.width, self.height)
        bytes_per_row = self.width * 4
        
        # Note: Using .tobytes() creates a CPU copy, but it's safe. 
        # For max performance, you'd use memoryview, but let's fix the leak first.
        self.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region,
            0,
            frame.data,
            bytes_per_row
        )
        
        return self.texture

    #Publish the Texture
    def publish_frame_texture(self, mtl_texture):
        self.syphon_server.publish_frame_texture(mtl_texture)

    #Convert CPU frame to GPU and publish it
    def publish_to_metal(self, frame):
        self.publish_frame_texture(self.numpy_to_metal(frame)) 
