import cv2
import numpy as np
import os
import OpenEXR
import Imath
from PIL import Image


# --- 4. Normalization Helpers ---
def normalize_channel(channel_data, min_val=None, max_val=None, target_range=(0, 1)):
        """Normalizes a single channel to a target range."""
        if channel_data.size == 0 or (np.all(channel_data == 0) and (min_val is None or max_val is None)):
            return np.zeros_like(channel_data, dtype=np.float32)

        # Handle NaNs and Infs
        channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)

        if min_val is None:
            min_val = channel_data.min()
        if max_val is None:
            max_val = channel_data.max()

        if max_val == min_val:
            return np.zeros_like(channel_data, dtype=np.float32)

        # Scale to 0-1 first
        normalized = (channel_data - min_val) / (max_val - min_val)

        # Then to target_range
        scaled = normalized * (target_range[1] - target_range[0]) + target_range[0]
        return scaled.astype(np.float32)



def load_exr_depth(exr_path):
    """Loads a single-channel float32 depth map from an EXR file."""
    try:
        file = OpenEXR.InputFile(exr_path)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Common depth channel names
        channel_names = ['Z', 'depth', 'Depth', 'z']
        depth_data = None
        for channel_name in channel_names:
            if channel_name in file.header()['channels']:
                data_type = Imath.PixelType(Imath.PixelType.FLOAT)
                z_slice = file.channels(channel_name, data_type)[0]
                depth_data = np.frombuffer(z_slice, dtype=np.float32).reshape(size[1], size[0])
                break

        if depth_data is None:
            print(
                f"Error: No recognized depth channel found in {exr_path}. Available channels: {file.header()['channels'].keys()}")
            return None

        # Handle potential infinite or NaN values in depth from LiDAR
        depth_data[np.isinf(depth_data)] = 0  # Set infinite depths to 0 or a max value
        depth_data[np.isnan(depth_data)] = 0  # Set NaN depths to 0

        return depth_data
    except Exception as e:
        print(f"Failed to load EXR {exr_path}: {e}")
        return None

def calculate_global_min_max(rgb_files,DEPTH_FRAMES_DIR):
  global_min_depth, global_max_depth = float('inf'), float('-inf')
  for i, rgb_filename in enumerate(rgb_files):
      base_filename = os.path.splitext(rgb_filename)[0]
      depth_filename = f"{base_filename}.exr"
      depth_path = os.path.join(DEPTH_FRAMES_DIR, depth_filename)

      if not os.path.exists(depth_path): continue
      depth_map = load_exr_depth(depth_path)
      if depth_map is None: continue

      valid_depths = depth_map[depth_map > 0]
      if valid_depths.size > 0:
          global_min_depth = min(global_min_depth, valid_depths.min())
          global_max_depth = max(global_max_depth, valid_depths.max())
  return global_min_depth, global_max_depth
