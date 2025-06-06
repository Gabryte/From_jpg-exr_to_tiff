import cv2
import numpy as np
import os
import OpenEXR
import Imath
from PIL import Image


# --- Normalization Helpers ---
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
#FOR GLOBAL DATASET MAX AND MIN
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


# --- Modified function for fixed max length  ---
def calculate_min_depth_with_fixed_max(rgb_files, DEPTH_FRAMES_DIR, fixed_max_depth_value):
    """
    Calculates the global minimum depth observed in the dataset,
    while using a predefined fixed maximum depth for normalization.

    Args:
        rgb_files (list): List of RGB filenames.
        DEPTH_FRAMES_DIR (str): Directory containing corresponding EXR depth files.
        fixed_max_depth_value (float): The maximum depth value to use for normalization
                                       (e.g., maximum achievable depth of the LiDAR sensor).

    Returns:
        tuple: (global_min_depth, fixed_max_depth_value)
    """
    global_min_depth = float('inf')

    for i, rgb_filename in enumerate(rgb_files):
        base_filename = os.path.splitext(rgb_filename)[0]
        depth_filename = f"{base_filename}.exr"
        depth_path = os.path.join(DEPTH_FRAMES_DIR, depth_filename)

        if not os.path.exists(depth_path):
            print(f"Skipping {depth_path}: not found.")
            continue

        depth_map = load_exr_depth(depth_path)
        if depth_map is None:
            print(f"Skipping {depth_path}: could not load depth.")
            continue

        # Filter out invalid depth values (0 or very large)
        # Assuming 0 means no depth data (background/invalid).
        # Also clip to the fixed_max_depth_value to only consider relevant data for min.
        valid_depths = depth_map[(depth_map > 0) & (depth_map <= fixed_max_depth_value)]

        if valid_depths.size > 0:
            global_min_depth = min(global_min_depth, valid_depths.min())

    # If no valid depths were found in the entire dataset, handle gracefully
    if global_min_depth == float('inf'):
        print("Warning: No valid depth data found in the dataset. Defaulting min depth to 0.")
        global_min_depth = 0.0

    return global_min_depth, fixed_max_depth_value
