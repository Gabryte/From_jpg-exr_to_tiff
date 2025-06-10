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
    """
    Loads a single-channel float32 depth map from an EXR file.
    Prioritizes 'Z', 'depth', etc., but falls back to 'R' if those are not found.
    Ensures the returned NumPy array is writable.
    """
    try:
        file = OpenEXR.InputFile(exr_path)
        dw = file.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # Prioritized depth channel names
        channel_names_priority = ['Z', 'depth', 'Depth', 'z']
        # Fallback channel names (e.g., if depth is encoded in Red)
        channel_names_fallback = ['R']

        depth_data_raw = None  # This variable will hold the *source* array, whether from priority or fallback
        data_type = Imath.PixelType(Imath.PixelType.FLOAT)

        available_channels = file.header()['channels'].keys()

        # Try to find a recognized depth channel first
        for channel_name in channel_names_priority:
            if channel_name in available_channels:
                z_slice = file.channels(channel_name, data_type)[0]
                depth_data_raw = np.frombuffer(z_slice, dtype=np.float32).reshape(size[1], size[0])
                print(f"Found depth in channel: '{channel_name}' for {exr_path}")
                break

        # If no priority channel found, try fallback channels
        if depth_data_raw is None:  # Only enter if no priority channel was found
            for channel_name in channel_names_fallback:
                if channel_name in available_channels:
                    z_slice = file.channels(channel_name, data_type)[0]
                    # --- FIX 1: Assign to depth_data_raw, not depth_data ---
                    depth_data_raw = np.frombuffer(z_slice, dtype=np.float32).reshape(size[1], size[0])
                    print(f"Falling back to channel: '{channel_name}' for depth in {exr_path}")
                    break  # Break after finding and loading the first fallback channel

        # --- FIX 2: Move this check *after* attempting both priority and fallback ---
        # If depth_data_raw is still None here, it means neither priority nor fallback channels were found.
        if depth_data_raw is None:
            print(f"Error: No recognized depth channel found in {exr_path}. Available channels: {available_channels}")
            return None

        # --- CRITICAL FIX 3: Make a writable copy *after* depth_data_raw has been assigned ---
        # This will now always be executed with a valid depth_data_raw (or the function would have returned)
        depth_data = depth_data_raw.copy()

        # Handle potential infinite or NaN values in depth from LiDAR
        depth_data[np.isinf(depth_data)] = 0  # Set infinite depths to 0 or a max value
        depth_data[np.isnan(depth_data)] = 0  # Set NaN depths to 0

        if np.any(np.isnan(depth_data)) or np.any(np.isinf(depth_data)):
            print(f"WARNING: NaNs/Infs still present after cleanup in {exr_path}!")

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

def find_missing_exr_for_jpg(jpg_directory, exr_directory):
    """
    Identifies JPG files in a directory that do not have a corresponding
    EXR file (with the same base name) in another directory.

    Args:
        jpg_directory (str): The path to the directory containing JPG files.
        exr_directory (str): The path to the directory containing EXR files.

    Returns:
        list: A list of full paths to JPG files that are missing their
              corresponding EXR file.
    """
    missing_jpg_files = []

    # Get all JPG filenames (without extension)
    jpg_base_names = set()
    for filename in os.listdir(jpg_directory):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            basename = os.path.splitext(filename)[0]
            jpg_base_names.add(basename)

    # Get all EXR filenames (without extension)
    exr_base_names = set()
    for filename in os.listdir(exr_directory):
        if filename.lower().endswith(".exr"):
            basename = os.path.splitext(filename)[0]
            exr_base_names.add(basename)

    # Find JPG basenames that are not in EXR basenames
    missing_basenames = jpg_base_names - exr_base_names

    # Reconstruct the full paths of the missing JPG files
    if missing_basenames:
        for filename in os.listdir(jpg_directory):
            current_basename = os.path.splitext(filename)[0]
            if current_basename in missing_basenames:
                missing_jpg_files.append(os.path.join(jpg_directory, filename))

    return missing_jpg_files


def check_exr_content(exr_file_path):
    #exr_file_path = "/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files/70_10.exr"

    #exr_file_path = '/home/jacobo/Desktop/Video RGB+D Florence/Annotated/2025-04-22--10-19-25/EXR_RGBD/depth/10.exr'

    try:
        file = OpenEXR.InputFile(exr_file_path)
        header = file.header()
        print(f"Header for {exr_file_path}:")
        for key, value in header.items():
            if key == 'channels':
                print(f"  Channels: {value.keys()}")  # Print just the channel names
            else:
                print(f"  {key}: {value}")

        # Example of how to read a specific channel (e.g., 'R')
        if 'R' in header['channels']:
            data_type = Imath.PixelType(Imath.PixelType.FLOAT)
            r_slice = file.channels('R', data_type)[0]
            dw = header['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            r_data = np.frombuffer(r_slice, dtype=np.float32).reshape(size[1], size[0])
            print(f"\nSuccessfully read 'R' channel. Min value: {r_data.min()}, Max value: {r_data.max()}")
            # You could also visualize r_data here to confirm it looks like depth
            # import matplotlib.pyplot as plt
            # plt.imshow(r_data, cmap='gray'); plt.colorbar(); plt.title("R Channel Data"); plt.show()
        else:
            print("No 'R' channel found.")

    except Exception as e:
        print(f"Error inspecting EXR file {exr_file_path}: {e}")

