import os

import numpy as np
from tqdm import tqdm

from lib.exr_functions import load_single_channel_exr_map


def calculate_log_depth_global_min_max(rgb_src_files_list, depth_src_dir):
    """
    Calculates the global minimum and maximum of log-transformed depth values
    across a collection of depth EXR files.

    This function iterates through a list of RGB image filenames, infers the
    corresponding depth EXR file paths, loads each EXR, applies a log1p
    transformation, and then finds the true minimum and maximum values
    across all valid (finite) pixels in all processed depth maps.
    It does it by updating global min/max without storing
    all log-depth values in memory, making it suitable for large datasets.

    Args:
        rgb_src_files_list (list): A list of strings, where each string is
                                   the base filename (e.g., 'image001.jpg')
                                   of an RGB image. These filenames are used
                                   to infer the names of the corresponding
                                   depth EXR files.
        depth_src_dir (str): The path to the directory containing all the
                             depth EXR files. It is assumed that for each
                             'image.jpg' in `rgb_src_files_list`, there is
                             a corresponding 'image.exr' in this directory.

    Returns:
        tuple: A tuple containing two floats:
               (global_min_log_depth, global_max_log_depth).
               These represent the absolute minimum and maximum log-transformed
               depth values encountered across the entire dataset.
               If no valid depth values are found, it defaults to (0.0, 1.0).
    """
    # Initialize global minimum log-depth to a very large positive number.
    # Any real log-depth value will be smaller than this, allowing correct initialization.
    global_min_log_depth = float('inf')
    # Initialize global maximum log-depth to a very small negative number.
    # Any real log-depth value will be larger than this, allowing correct initialization.
    global_max_log_depth = float('-inf')

    print("\nStarting calculation of global min/max log-depth for consistent normalization...")
    # Flag to track if at least one valid depth map with finite log-depth values was processed.
    # This helps in handling cases where the entire dataset might be empty or contain only invalid depths.
    found_valid_depth = False

    # Iterate through each RGB filename provided. tqdm adds a progress bar to the loop.
    for rgb_filename in tqdm(rgb_src_files_list, desc="Scanning depths for global range"):
        # Extract the base filename (e.g., 'image001' from 'image001.jpg').
        base_filename = os.path.splitext(rgb_filename)[0]
        # Construct the expected filename for the corresponding EXR depth map.
        depth_filename = f"{base_filename}.exr"
        # Construct the full path to the EXR depth map.
        depth_path = os.path.join(depth_src_dir, depth_filename)

        # Check if the constructed depth EXR file actually exists on disk.
        if os.path.exists(depth_path):
            # Load the single-channel EXR depth map into a NumPy array.
            # `load_single_channel_exr_map` is assumed to handle EXR reading.
            depth_map = load_single_channel_exr_map(depth_path)

            # Proceed only if the depth map was loaded successfully (not None).
            if depth_map is not None:
                # Apply the log1p transformation: log(1 + x).
                # This is commonly used for depth data to compress its range,
                # especially since depth values can span a wide range and often
                # have a long tail distribution (many small values, few large ones).
                # Adding 1 handles zero depth values gracefully (log(1) = 0).
                log_depth_map = np.log1p(depth_map)

                # Filter out any non-finite values (NaNs, positive/negative Infs)
                # from the log-transformed depth map. These values can arise from
                # invalid depth measurements (e.g., background, sensor noise)
                # or from log(0) if not using log1p.
                log_depth_map_finite = log_depth_map[np.isfinite(log_depth_map)]

                # If there are any finite (valid) log-depth values remaining after filtering:
                if log_depth_map_finite.size > 0:
                    # Find the minimum and maximum log-depth values within the current map.
                    current_min = log_depth_map_finite.min()
                    current_max = log_depth_map_finite.max()

                    # Update the global minimum and maximum log-depth values.
                    # This ensures we track the absolute min/max across all processed maps.
                    global_min_log_depth = min(global_min_log_depth, current_min)
                    global_max_log_depth = max(global_max_log_depth, current_max)
                    # Set the flag to True, indicating at least one valid depth map contributed to the range.
                    found_valid_depth = True
                # else: No valid finite values in this specific depth map, skip its contribution.
            # else: Depth map could not be loaded, already handled by load_single_channel_exr_map.
        # else: Depth EXR file does not exist, it will be skipped automatically.

    # --- Post-processing and Fallback Logic ---
    # After iterating through all files, check if any valid depth values were found.
    if not found_valid_depth:
        # If `found_valid_depth` is still False, it means no EXR files were found,
        # or all found EXR files either failed to load or contained only non-finite values.
        print("Warning: No valid log-depth values were found across the entire dataset. "
              "This may indicate missing EXR files, loading issues, or entirely invalid depth data. "
              "Using a default range of 0.0-1.0 for normalization to prevent errors downstream.")
        global_min_log_depth = 0.0
        global_max_log_depth = 1.0
    # This `elif` acts as an additional safeguard. It should ideally be covered by `found_valid_depth` check,
    # but in very rare or unexpected edge cases where `found_valid_depth` somehow got set but min/max
    # remained at their initial infinite values (e.g., if only very specific non-finite values were
    # encountered and `np.isfinite` missed them for some reason, or an empty `log_depth_map_finite`
    # passed through previous checks, which shouldn't happen with `size > 0`).
    elif global_min_log_depth == float('inf') or global_max_log_depth == float('-inf'):
        print("Warning: Global min/max values remained at initial infinity/negative infinity after processing. "
              "This suggests the dataset might contain only invalid or unprocessable depth values even after initial filtering. "
              "Using a default range of 0.0-1.0 for normalization.")
        global_min_log_depth = 0.0
        global_max_log_depth = 1.0

    # Print the final calculated global min and max log-depth values.
    print(f"Calculation complete. Global Min Log Depth: {global_min_log_depth}, "
          f"Global Max Log Depth: {global_max_log_depth}")

    return global_min_log_depth, global_max_log_depth
