import shutil

import numpy as np
import hashlib
import os


def normalize_array_to_range(data_array, min_val=None, max_val=None, target_range=(0, 1)):
    """
    Scales a NumPy array to a specified target range.

    This function performs min-max normalization. If the minimum and maximum values
    of the data are not provided, they are computed from the array itself. It is
    designed to robustly handle edge cases such as empty arrays, arrays with
    uniform values, and non-finite numbers (NaN, infinity).

    Args:
        data_array (np.ndarray): The input NumPy array to be normalized.
        min_val (float, optional): The minimum value to use for normalization.
                                   If None, the array's minimum is used. Defaults to None.
        max_val (float, optional): The maximum value to use for normalization.
                                   If None, the array's maximum is used. Defaults to None.
        target_range (tuple[float, float]): The desired output range as (min, max).
                                            Defaults to (0, 1).

    Returns:
        np.ndarray: The normalized array with dtype float32, with values scaled
                    to the specified target_range.
    """
    # Handle the edge case of an empty array to prevent errors.
    if data_array.size == 0:
        return np.zeros_like(data_array, dtype=np.float32)

    # Replace non-finite values (NaN, inf) with 0.0 to ensure calculations are valid.
    data_array_finite = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Determine the min and max values from the data if not provided by the user.
    if min_val is None:
        min_val = data_array_finite.min()
    if max_val is None:
        max_val = data_array_finite.max()

    # If all values in the array are the same, division by zero would occur.
    # To handle this, return an array where all values are the midpoint of the target range.
    if max_val == min_val:
        mid_point = (target_range[0] + target_range[1]) / 2.0
        return np.full_like(data_array, mid_point, dtype=np.float32)

    # Perform min-max normalization: first scale the data to the [0, 1] range.
    normalized = (data_array_finite - min_val) / (max_val - min_val)

    # Scale and shift the [0, 1] result to the desired target_range.
    scale_factor = target_range[1] - target_range[0]
    scaled = normalized * scale_factor + target_range[0]

    # Clip the values to ensure they fall strictly within the target range and set the final data type.
    return np.clip(scaled, target_range[0], target_range[1]).astype(np.float32)


def generate_file_hash(filepath, hash_algorithm='md5'):
    """
    Generates a hexadecimal hash for a file's content.

    This function reads the file in binary chunks to efficiently handle large
    files without consuming excessive memory. It supports any hashing algorithm
    available in Python's `hashlib` module.

    Args:
        filepath (str): The absolute or relative path to the file.
        hash_algorithm (str): The name of the hash algorithm to use (e.g.,
                              'md5', 'sha256'). Defaults to 'md5'.

    Returns:
        str: The hexadecimal representation of the file's hash.
             Returns an empty string if the file cannot be read.
    """
    try:
        # Initialize the hasher with the specified algorithm.
        hasher = hashlib.new(hash_algorithm)

        # Open the file in binary read mode ('rb'). This is crucial for hashing
        # as it ensures consistent byte reading across all platforms.
        with open(filepath, 'rb') as f:
            # Read the file in fixed-size chunks (e.g., 8KB) in a loop.
            # This is memory-efficient for very large files.
            while True:
                chunk = f.read(8192)
                if not chunk:
                    # End of file has been reached.
                    break
                # Update the hasher with the content of the current chunk.
                hasher.update(chunk)

        # Return the final hash as a hexadecimal string.
        return hasher.hexdigest()
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return ""


def get_input_directories(directories_path):
    """
    Finds all immediate subdirectories within a given path and appends a
    specific child directory name to each.

    For example, if `directories_path` contains `folder1` and `folder2`, this
    function will return `['.../folder1/EXR_RGBD', '.../folder2/EXR_RGBD']`.

    Args:
        directories_path (str): The path to the parent directory to scan.

    Returns:
        list[str]: A list of complete paths to the "EXR_RGBD" subfolders.
    """
    # Use os.scandir for better performance than os.listdir, as it fetches
    # file type information during the initial directory scan.
    subfolders = [f.path for f in os.scandir(directories_path) if f.is_dir()]

    # Iterate through the list of found subfolders to modify each path.
    for i, folder in enumerate(subfolders):
        # Append the target subfolder name using os.path.join for
        # cross-platform compatibility (handles '/' vs '\' correctly).
        subfolders[i] = os.path.join(folder, "EXR_RGBD")

    return subfolders

def copy_and_rename_file(source_directory, destination_directory, original_filename, new_filename):
    """
    Copies a file to a new location and gives it a new name.

    This utility ensures the source file exists before attempting the copy and
    creates the destination directory if it doesn't already exist. It uses
    `shutil.copy2` to preserve file metadata (e.g., timestamps).

    Args:
        source_directory (str): The path to the directory containing the original file.
        destination_directory (str): The path to the target directory.
        original_filename (str): The name of the file to be copied.
        new_filename (str): The desired new name for the file at the destination.
    """
    # Construct full, platform-agnostic paths for the source and destination files.
    source_path = os.path.join(source_directory, original_filename)
    destination_path = os.path.join(destination_directory, new_filename)

    # Pre-flight check: ensure the source file actually exists before proceeding.
    if not os.path.exists(source_path):
        print(f"Error: Source file '{source_path}' does not exist.")
        return

    # Ensure the destination directory exists. `exist_ok=True` prevents an
    # error if the directory has already been created.
    os.makedirs(destination_directory, exist_ok=True)

    try:
        # Use shutil.copy2 to copy the file. Unlike shutil.copy, copy2 also
        # attempts to preserve all file metadata.
        shutil.copy2(source_path, destination_path)
        print(f"Successfully copied '{original_filename}' to '{destination_path}'.")
    except Exception as e:
        # Catch any exceptions during the file copy operation for graceful error handling.
        print(f"An error occurred while copying the file: {e}")

# def visualize_multichannel_tiff(tiff_file_path: str, preview_output_dir: str):
#     """
#     Loads a 4-channel (RGB + Normalized Depth) TIFF in CHW format,
#     displays its individual channels, and saves preview images.
#
#     Args:
#         tiff_file_path (str): The full path to the TIFF file to visualize.
#         preview_output_dir (str): The directory where the preview images
#                                   (RGB and Depth) should be saved.
#     """
#     try:
#         # Ensure the preview output directory exists
#         os.makedirs(preview_output_dir, exist_ok=True)
#
#         # Load the TIFF image
#         stacked_channels_chw = tifffile.imread(tiff_file_path)
#
#         print(f"Loaded TIFF: {tiff_file_path}")
#         print(f"Shape (C, H, W): {stacked_channels_chw.shape}")
#         print(f"Data Type: {stacked_channels_chw.dtype}")
#
#         if stacked_channels_chw.shape[0] != 4:
#             print(
#                 f"Error: Expected 4 channels (RGB + Depth), but found {stacked_channels_chw.shape[0]} in {tiff_file_path}.")
#             return
#
#         # Extract RGB channels (channels 0, 1, 2) and transpose to HWC
#         rgb_hwc = stacked_channels_chw[0:3, :, :].transpose((1, 2, 0))  # (H, W, C)
#
#         # Extract Normalized Depth channel (channel 3)
#         normalized_depth = stacked_channels_chw[3, :, :]  # (H, W)
#
#         # --- Display Images using Matplotlib ---
#         plt.figure(figsize=(15, 7))
#
#         # Plot RGB
#         plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
#         plt.imshow(rgb_hwc)  # matplotlib handles float 0-1 RGB
#         plt.title('RGB Channels (0-1 Normalized)')
#         plt.axis('off')
#
#         # Plot Normalized Depth with a colormap
#         plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
#         depth_img = plt.imshow(normalized_depth, cmap='magma')
#         plt.colorbar(depth_img, label='Normalized Depth (0-1)')
#         plt.title('Normalized Depth Channel')
#         plt.axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#         # --- Save Individual Channel Previews ---
#         # Get base filename without extension for saving previews
#         base_name = os.path.splitext(os.path.basename(tiff_file_path))[0]
#
#         # Save RGB preview
#         rgb_preview = (rgb_hwc * 255).astype(np.uint8)
#         cv2.imwrite(os.path.join(preview_output_dir, f"{base_name}_rgb_preview.png"),
#                     cv2.cvtColor(rgb_preview, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV save
#         print(f"Saved RGB preview to {os.path.join(preview_output_dir, f'{base_name}_rgb_preview.png')}")
#
#         # Save Depth preview (with colormap)
#         depth_for_colormap = (normalized_depth * 255).astype(np.uint8)
#         depth_colored_preview = cv2.applyColorMap(depth_for_colormap, cv2.COLORMAP_MAGMA)
#         cv2.imwrite(os.path.join(preview_output_dir, f"{base_name}_depth_preview.png"), depth_colored_preview)
#         print(f"Saved Depth preview to {os.path.join(preview_output_dir, f'{base_name}_depth_preview.png')}")
#
#     except Exception as e:
#         print(f"An error occurred while visualizing {tiff_file_path}: {e}")
#


# def load_exr_depth(exr_path):
#     """
#     Loads a single-channel float32 depth map from an EXR file.
#     Prioritizes 'Z', 'depth', etc., but falls back to 'R' if those are not found.
#     Ensures the returned NumPy array is writable.
#     """
#     try:
#         file = OpenEXR.InputFile(exr_path)
#         dw = file.header()['dataWindow']
#         size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#
#         # Prioritized depth channel names
#         #channel_names_priority = ['Z', 'depth', 'Depth', 'z']
#         # Fallback channel names (e.g., if depth is encoded in Red)
#         channel_names_fallback = ['R']
#
#         depth_data_raw = None  # This variable will hold the *source* array, whether from priority or fallback
#         data_type = Imath.PixelType(Imath.PixelType.FLOAT)
#
#         available_channels = file.header()['channels'].keys()
#
#         # Try to find a recognized depth channel first
#         #for channel_name in channel_names_priority:
#             #if channel_name in available_channels:
#                 #z_slice = file.channels(channel_name, data_type)[0]
#                 #depth_data_raw = np.frombuffer(z_slice, dtype=np.float32).reshape(size[1], size[0])
#                 #print(f"Found depth in channel: '{channel_name}' for {exr_path}")
#                 #break
#
#         # If no priority channel found, try fallback channels
#         if depth_data_raw is None:  # Only enter if no priority channel was found
#             for channel_name in channel_names_fallback:
#                 if channel_name in available_channels:
#                     z_slice = file.channels(channel_name, data_type)[0]
#                     # --- FIX 1: Assign to depth_data_raw, not depth_data ---
#                     depth_data_raw = np.frombuffer(z_slice, dtype=np.float32).reshape(size[1], size[0])
#                     print(f"Falling back to channel: '{channel_name}' for depth in {exr_path}")
#                     break  # Break after finding and loading the first fallback channel
#
#         # --- FIX 2: Move this check *after* attempting both priority and fallback ---
#         # If depth_data_raw is still None here, it means neither priority nor fallback channels were found.
#         if depth_data_raw is None:
#             print(f"Error: No recognized depth channel found in {exr_path}. Available channels: {available_channels}")
#             return None
#
#         # --- CRITICAL FIX 3: Make a writable copy *after* depth_data_raw has been assigned ---
#         # This will now always be executed with a valid depth_data_raw (or the function would have returned)
#         depth_data = depth_data_raw.copy()
#
#         # Handle potential infinite or NaN values in depth from LiDAR
#         depth_data[np.isinf(depth_data)] = 0  # Set infinite depths to 0 or a max value
#         depth_data[np.isnan(depth_data)] = 0  # Set NaN depths to 0
#
#         if np.any(np.isnan(depth_data)) or np.any(np.isinf(depth_data)):
#             print(f"WARNING: NaNs/Infs still present after cleanup in {exr_path}!")
#
#         return depth_data
#     except Exception as e:
#         print(f"Failed to load EXR {exr_path}: {e}")
#         return None

# def normalize_channel(channel_data, min_val=None, max_val=None, target_range=(0, 1)):
#         """Normalizes a single channel to a target range."""
#         if channel_data.size == 0 or (np.all(channel_data == 0) and (min_val is None or max_val is None)):
#             return np.zeros_like(channel_data, dtype=np.float32)
#
#         # Handle NaNs and Infs
#         channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
#
#         if min_val is None:
#             min_val = channel_data.min()
#         if max_val is None:
#             max_val = channel_data.max()
#
#         if max_val == min_val:
#             return np.zeros_like(channel_data, dtype=np.float32)
#
#         # Scale to 0-1 first
#         normalized = (channel_data - min_val) / (max_val - min_val)
#
#         # Then to target_range
#         scaled = normalized * (target_range[1] - target_range[0]) + target_range[0]
#         return scaled.astype(np.float32)
