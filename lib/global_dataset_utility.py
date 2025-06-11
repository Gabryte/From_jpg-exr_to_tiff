

import cv2

from lib.utility import load_exr_depth
import os
import random
import uuid


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


def down_grade_resolution_in_four_thirds(TARGET_WIDTH,rgb_frame,depth_map):
    print("Automatically rescale in a 4:3 format...")
    TARGET_HEIGHT = int((TARGET_WIDTH / 4) * 3)
    print(f"New resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    # --- Resizing ALL input channels to TARGET_WIDTH x TARGET_HEIGHT ---
    # RGB (uint8)
    rgb_frame_resized = cv2.resize(rgb_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Depth map (float32)
    # Note: INTER_AREA is good for downscaling, INTER_LINEAR or INTER_CUBIC for upscaling or general resizing.
    # Since depth maps are often continuous, LINEAR might be slightly better than AREA even for downscaling.
    depth_map_resized = cv2.resize(depth_map, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return rgb_frame_resized, depth_map_resized

def split_80_20_training_dataset_for_validation_and_shuffle_them(TRAIN_PATH,TRAIN_LABEL_PATH,VALIDATION_PATH,VALIDATION_LABEL_PATH):


def shuffle_frames_randomly(IMAGES_PATH,LABEL_PATH):
    """
    Shuffles image and corresponding label files randomly within a YOLOv5 dataset.

    This function renames image files and their linked label files to maintain
    the consistency of the dataset after shuffling. It uses a two-pass renaming
    strategy to avoid name conflicts during the shuffling process.

    Args:
        IMAGES_PATH (str): The path to the directory containing image files.
        LABEL_PATH (str): The path to the directory containing label files (TXT).
    """
    # Validate that the provided paths exist and are directories
    if not os.path.isdir(IMAGES_PATH):
        print(f"Error: IMAGES_PATH '{IMAGES_PATH}' does not exist or is not a directory.")
        return
    if not os.path.isdir(LABEL_PATH):
        print(f"Error: LABEL_PATH '{LABEL_PATH}' does not exist or is not a directory.")
        return

    # Define supported image extensions for filtering
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    # 1. Gather all image and label files from the specified directories
    # os.path.isfile() is used to ensure we only process actual files, not subdirectories
    # Added filters to exclude hidden files (starting with '.')
    image_files = []
    for f in os.listdir(IMAGES_PATH):
        full_path = os.path.join(IMAGES_PATH, f)
        if os.path.isfile(full_path) and f.lower().endswith(image_extensions):
            if f.startswith('.'):  # Exclude files starting with a dot, typically hidden on Linux
                print(f"Skipping hidden file: '{f}'")
                continue
            image_files.append(f)

    label_files = [f for f in os.listdir(LABEL_PATH) if
                   os.path.isfile(os.path.join(LABEL_PATH, f)) and f.lower().endswith('.txt')]

    # Create a dictionary for quick lookup of label files by their base name (e.g., 'frame001')
    labels_dict = {os.path.splitext(f)[0]: f for f in label_files}

    # List to store tuples: (original_image_full_path, original_label_full_path, original_image_extension)
    # This list will hold information for all successfully matched image-label pairs.
    matched_pairs_info = []

    # Iterate through image files to find their corresponding label files
    for img_file in image_files:
        base_name_without_ext = os.path.splitext(img_file)[0]  # Get filename without extension
        img_ext = os.path.splitext(img_file)[1]  # Get the original image extension (e.g., '.jpg')

        # Check if a label file with the same base name exists
        if base_name_without_ext in labels_dict:
            original_image_path = os.path.join(IMAGES_PATH, img_file)
            original_label_path = os.path.join(LABEL_PATH, labels_dict[base_name_without_ext])
            matched_pairs_info.append((original_image_path, original_label_path, img_ext))
        else:
            print(f"Warning: No matching label (.txt) found for image: '{img_file}'. Skipping this pair.")

    # If no matched pairs are found, exit the function
    if not matched_pairs_info:
        print("No matched image-label pairs found with corresponding .txt files. Exiting.")
        return

    print(f"Found {len(matched_pairs_info)} image-label pairs to shuffle.")

    # This list will store (temporary_image_full_path, temporary_label_full_path, original_image_extension)
    # for files that were successfully renamed in the first pass.
    temp_paths_after_first_pass = []

    # 2. First Pass: Rename all original files to unique temporary names
    # This avoids potential conflicts if a new shuffled name happens to be an existing original name.
    print("\n--- Phase 1: Renaming original files to temporary unique names ---")
    for i, (orig_img_path, orig_lbl_path, img_ext) in enumerate(matched_pairs_info):
        temp_id = uuid.uuid4().hex  # Generate a unique hexadecimal ID for temporary names

        temp_img_name = f"{temp_id}{img_ext}"  # e.g., 'a1b2c3d4e5f6.jpg'
        temp_lbl_name = f"{temp_id}.txt"  # e.g., 'a1b2c3d4e5f6.txt'

        temp_img_path = os.path.join(IMAGES_PATH, temp_img_name)
        temp_lbl_path = os.path.join(LABEL_PATH, temp_lbl_name)

        try:
            # Attempt to rename the original image and label files to their temporary names
            os.rename(orig_img_path, temp_img_path)
            os.rename(orig_lbl_path, temp_lbl_path)
            # If successful, add the temporary paths and original extension to our list
            temp_paths_after_first_pass.append((temp_img_path, temp_lbl_path, img_ext))
        except FileNotFoundError:
            print(
                f"Error: Original file not found during temp rename for '{os.path.basename(orig_img_path)}' or its label. Skipping this pair.")
        except Exception as e:
            # Catch any other unexpected errors during renaming
            print(
                f"An unexpected error occurred during temporary rename of '{os.path.basename(orig_img_path)}': {e}. Skipping this pair.")

    # If no files were successfully moved to temporary names, there's nothing to shuffle
    if not temp_paths_after_first_pass:
        print("No files were successfully renamed to temporary names. Shuffling aborted.")
        return

    # 3. Shuffle the list of temporary file paths
    # This randomizes the order in which files will be renamed in the final pass.
    print("\n--- Phase 2: Shuffling temporary file paths ---")
    random.shuffle(temp_paths_after_first_pass)
    print("Temporary file paths have been randomly shuffled.")

    # 4. Second Pass: Rename from temporary names to new shuffled sequential names
    # This assigns new, organized names to the now-shuffled files.
    print("\n--- Phase 3: Renaming temporary files to final shuffled sequential names ---")

    # Determine the number of digits needed for zero-padding in the new sequential names (e.g., '001' for 3 digits)
    # This ensures consistent naming like 'frame_000', 'frame_001', up to 'frame_999' for 1000 files.
    num_digits = len(str(len(
        temp_paths_after_first_pass) - 1))  # If 1 file, this is len(str(0))=1. If 10 files, len(str(9))=1. If 100 files, len(str(99))=2.
    if num_digits == 0:  # Handle edge case where there might be 0 or 1 file resulting in num_digits = 0
        num_digits = 1

    for i, (temp_img_path, temp_lbl_path, img_ext) in enumerate(temp_paths_after_first_pass):
        # Create a new sequential base name (e.g., 'frame_000', 'frame_001', etc.)
        new_base_name = f"frame_{i:0{num_digits}d}"

        # Construct the final full paths for the shuffled image and label files
        final_img_path = os.path.join(IMAGES_PATH, f"{new_base_name}{img_ext}")
        final_lbl_path = os.path.join(LABEL_PATH, f"{new_base_name}.txt")

        try:
            # Attempt to rename the temporary image and label files to their final shuffled names
            os.rename(temp_img_path, final_img_path)
            os.rename(temp_lbl_path, final_lbl_path)
        except FileNotFoundError:
            print(
                f"Error: Temporary file not found during final rename for '{os.path.basename(temp_img_path)}' or its label. This file might have been skipped earlier. Skipping.")
        except Exception as e:
            # Catch any other unexpected errors during the final renaming
            print(
                f"An unexpected error occurred during final rename of '{os.path.basename(temp_img_path)}': {e}. Skipping this pair.")

    print("\n--- Shuffling complete! ---")
    print("All image and corresponding label files have been randomly shuffled and renamed sequentially.")






