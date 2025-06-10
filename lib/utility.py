import shutil

import OpenEXR
import Imath
import numpy as np
import hashlib
import os

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

def generate_file_hash(filepath, hash_algorithm='md5'):
    """Generates the hash of a file's content."""
    hasher = hashlib.new(hash_algorithm)
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)  # Read file in chunks
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_input_directories():
   home = os.path.expanduser("~")
   directories_path = os.path.join(home,"Desktop","Video RGB+D Florence","Annotated")
   subfolders = [f.path for f in os.scandir(directories_path) if f.is_dir()]
   for i,folder in enumerate(subfolders):
       subfolders[i] = os.path.join(folder,"EXR_RGBD")
   return subfolders


def copy_and_rename_file(source_directory, destination_directory, original_filename, new_filename):
    """
    Copies a file from a source directory to a destination directory and renames it.

    Args:
        source_directory (str): The path to the directory where the original file is located.
        destination_directory (str): The path to the directory where the file will be copied.
        original_filename (str): The original name of the file (including extension).
        new_filename (str): The new name for the copied file (including extension).
    """
    source_path = os.path.join(source_directory, original_filename)
    destination_path = os.path.join(destination_directory, new_filename)

    # Ensure the source file exists
    if not os.path.exists(source_path):
        print(f"Error: Source file '{source_path}' does not exist.")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory, exist_ok=True)

    try:
        shutil.copy2(source_path, destination_path)
        print(f"Successfully copied '{original_filename}' from '{source_directory}' to '{destination_directory}' as '{new_filename}'.")
    except Exception as e:
        print(f"An error occurred while copying the file: {e}")


