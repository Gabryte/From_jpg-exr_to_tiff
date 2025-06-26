import shutil
from PIL import Image # Import Pillow for image processing
from lib.utility import load_exr_depth, load_single_channel_exr_map
import random
import uuid
import cv2
import numpy as np
import os
from tqdm import tqdm
import tifffile  # Use tifffile as it's best for multi-channel TIFFs, even if converting to PNG later


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

# --- REFACTORED HELPER FUNCTION FOR THE NEW DIRECT CONVERSION FUNCTION BYPASSING THE TIFF DATASET---
def calculate_log_depth_global_min_max(rgb_src_files_list, depth_src_dir):
    """
    Calculates global min/max of log-transformed depth values across a list of RGB files
    and their corresponding depth EXR files, without storing all values in memory.
    """
    global_min_log_depth = float('inf')  # Initialize with positive infinity
    global_max_log_depth = float('-inf') # Initialize with negative infinity

    print("\nCalculating global min/max log-depth for consistent normalization...")
    found_valid_depth = False # Flag to check if any valid depth was found

    for rgb_filename in tqdm(rgb_src_files_list, desc="Scanning depths for global range"):
        base_filename = os.path.splitext(rgb_filename)[0]
        depth_filename = f"{base_filename}.exr"
        depth_path = os.path.join(depth_src_dir, depth_filename)

        if os.path.exists(depth_path):
            depth_map = load_single_channel_exr_map(depth_path)
            if depth_map is not None:
                # Apply log1p transformation
                log_depth_map = np.log1p(depth_map)

                # Filter out any non-finite values (NaNs, Infs)
                log_depth_map_finite = log_depth_map[np.isfinite(log_depth_map)]

                if log_depth_map_finite.size > 0:
                    current_min = log_depth_map_finite.min()
                    current_max = log_depth_map_finite.max()

                    # Update global min/max
                    global_min_log_depth = min(global_min_log_depth, current_min)
                    global_max_log_depth = max(global_max_log_depth, current_max)
                    found_valid_depth = True # At least one valid depth map processed

    if not found_valid_depth:
        # Fallback if no valid depths were processed at all
        print("Warning: No valid log-depth values found across the dataset. Using default 0-1 range for normalization.")
        global_min_log_depth = 0.0
        global_max_log_depth = 1.0
    elif global_min_log_depth == float('inf') or global_max_log_depth == float('-inf'):
        # This case should ideally be caught by found_valid_depth = False,
        # but as a failsafe if an entire dataset consists only of filtered-out values.
        print("Warning: Global min/max values remained at initial infinity/negative infinity. Dataset might contain only invalid depths after filtering. Using default 0-1 range.")
        global_min_log_depth = 0.0
        global_max_log_depth = 1.0


    print(f"Global Min Log Depth: {global_min_log_depth}, Global Max Log Depth: {global_max_log_depth}")
    return global_min_log_depth, global_max_log_depth

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


def resize_resolution_maintaining_aspect_ratio(TARGET_WIDTH, IMAGES_PATH):
    """
    Resizes all images in a specified directory to a target width while maintaining
    their aspect ratio. Images are overwritten with their resized versions.
    It handles both downgrading and upgrading resolution. Hidden files (starting with '.')
    are skipped.

    Args:
        TARGET_WIDTH (int): The desired width for all images.
        IMAGES_PATH (str): The path to the directory containing image files.
    """
    # Validate IMAGES_PATH
    if not os.path.isdir(IMAGES_PATH):
        print(f"Error: IMAGES_PATH '{IMAGES_PATH}' does not exist or is not a directory.")
        return

    # Validate TARGET_WIDTH
    if not isinstance(TARGET_WIDTH, int) or TARGET_WIDTH <= 0:
        print(f"Error: TARGET_WIDTH must be a positive integer. Got '{TARGET_WIDTH}'.")
        return

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    resized_count = 0

    print(f"\n--- Resizing images in '{IMAGES_PATH}' to a target width of {TARGET_WIDTH}px ---")

    for filename in os.listdir(IMAGES_PATH):
        full_path = os.path.join(IMAGES_PATH, filename)

        # Skip directories and hidden files
        if os.path.isdir(full_path) or filename.startswith('.'):
            # print(f"Skipping directory or hidden file: '{filename}'")
            continue

        # Check if it's a supported image file
        if not filename.lower().endswith(image_extensions):
            print(f"Skipping non-image file: '{filename}'")
            continue

        try:
            with Image.open(full_path) as img:
                original_width, original_height = img.size

                # Calculate new height maintaining aspect ratio
                aspect_ratio = original_height / original_width
                new_height = int(TARGET_WIDTH * aspect_ratio)

                # Resize the image using Image.Resampling.LANCZOS
                resized_img = img.resize((TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)

                # Save the resized image, overwriting the original
                resized_img.save(full_path)
                resized_count += 1
                # print(f"Resized '{filename}' from {original_width}x{original_height} to {TARGET_WIDTH}x{new_height}.")

        except FileNotFoundError:
            print(f"Error: Image file not found during resize operation for '{filename}'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{filename}': {e}. Skipping.")

    print(f"\n--- Image resizing complete! ---")
    print(f"Successfully resized {resized_count} images in '{IMAGES_PATH}'.")

# --- REFACTORED HELPER FUNCTION FOR THE NEW DIRECT CONVERSION FUNCTION BYPASSING THE TIFF DATASET---
def resize_rgb_and_depth_maintain_aspect_ratio(TARGET_WIDTH, rgb_frame, depth_map):
    """
    Resizes RGB and depth frames to a target width while maintaining their original aspect ratio.
    """
    original_height, original_width = rgb_frame.shape[0], rgb_frame.shape[1]

    # Calculate target height to maintain aspect ratio
    target_height = int(original_height * (TARGET_WIDTH / original_width))

    print(f"Resizing to: {TARGET_WIDTH}x{target_height} (maintaining aspect ratio)")

    # Resize RGB (uint8)
    rgb_frame_resized = cv2.resize(rgb_frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)

    # Resize Depth map (float32)
    depth_map_resized = cv2.resize(depth_map, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_LINEAR)

    return rgb_frame_resized, depth_map_resized


def downgrade_resolution_in_four_thirds_with_depths(TARGET_WIDTH, rgb_frame, depth_map):
    print("Automatically rescale in a 4:3 format...")
    TARGET_HEIGHT = int((TARGET_WIDTH * 4) / 3)
    print(f"New resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    # --- Resizing ALL input channels to TARGET_WIDTH x TARGET_HEIGHT ---
    # RGB (uint8)
    rgb_frame_resized = cv2.resize(rgb_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Depth map (float32)
    # Note: INTER_AREA is good for downscaling, INTER_LINEAR or INTER_CUBIC for upscaling or general resizing.
    # Since depth maps are often continuous, LINEAR might be slightly better than AREA even for downscaling.
    depth_map_resized = cv2.resize(depth_map, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

    return rgb_frame_resized, depth_map_resized


def shuffle_frames_randomly(IMAGES_PATH, LABEL_PATH):
    """
    Shuffles all image files within IMAGES_PATH, and if a corresponding
    label file exists in LABEL_PATH, that label file is also shuffled
    alongside its image. Images without a corresponding label are still
    shuffled and renamed.

    This function renames files using a two-pass strategy to avoid conflicts.
    It excludes hidden files (starting with '.').

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

    # 1. Gather all valid image files and identify their potential label partners
    all_files_to_process = []  # Stores (original_image_full_path, original_label_full_path_or_None, original_image_extension)

    # Get all label files and create a dictionary for quick lookup by base name
    label_files_in_dir = [f for f in os.listdir(LABEL_PATH) if
                          os.path.isfile(os.path.join(LABEL_PATH, f)) and f.lower().endswith('.txt')]
    labels_dict = {os.path.splitext(f)[0]: os.path.join(LABEL_PATH, f) for f in label_files_in_dir}

    # Iterate through image files to determine if they have a corresponding label
    for img_file in os.listdir(IMAGES_PATH):
        full_img_path = os.path.join(IMAGES_PATH, img_file)

        # Skip directories and hidden files
        if os.path.isdir(full_img_path) or img_file.startswith('.'):
            # print(f"Skipping directory or hidden file: '{img_file}'")
            continue

        # Check if it's a supported image file
        if not img_file.lower().endswith(image_extensions):
            print(f"Skipping non-image file: '{img_file}' in IMAGES_PATH.")
            continue

        base_name_without_ext = os.path.splitext(img_file)[0]
        img_ext = os.path.splitext(img_file)[1]

        # Check if a label file exists for this image
        corresponding_label_path = labels_dict.get(base_name_without_ext)

        # Add to the list to be processed, even if no label exists
        all_files_to_process.append((full_img_path, corresponding_label_path, img_ext))
        if corresponding_label_path is None:
            print(f"Info: Image '{img_file}' has no corresponding label file. It will still be renamed.")

    if not all_files_to_process:
        print("No valid image files found to shuffle. Exiting.")
        return

    print(f"Found {len(all_files_to_process)} image files (some without labels) to shuffle.")

    # This list will store (temporary_image_full_path, temporary_label_full_path_or_None, original_image_extension)
    # for files that were successfully renamed in the first pass.
    temp_paths_after_first_pass = []

    # 2. First Pass: Rename all original files to unique temporary names
    # This avoids potential conflicts if a new shuffled name happens to be an existing original name.
    print("\n--- Phase 1: Renaming original files to temporary unique names ---")
    for i, (orig_img_path, orig_lbl_path, img_ext) in enumerate(all_files_to_process):
        temp_id = uuid.uuid4().hex  # Generate a unique hexadecimal ID for temporary names

        temp_img_name = f"{temp_id}{img_ext}"
        temp_img_path = os.path.join(IMAGES_PATH, temp_img_name)

        temp_lbl_name = f"{temp_id}.txt"
        temp_lbl_path = os.path.join(LABEL_PATH, temp_lbl_name) if orig_lbl_path else None

        try:
            os.rename(orig_img_path, temp_img_path)
            if orig_lbl_path:  # Only rename label if it existed
                os.rename(orig_lbl_path, temp_lbl_path)
            temp_paths_after_first_pass.append((temp_img_path, temp_lbl_path, img_ext))
        except FileNotFoundError:
            print(
                f"Error: Original file not found during temp rename for '{os.path.basename(orig_img_path)}' (or its label). Skipping this pair.")
        except Exception as e:
            print(
                f"An unexpected error occurred during temporary rename of '{os.path.basename(orig_img_path)}': {e}. Skipping this pair.")

    if not temp_paths_after_first_pass:
        print("No files were successfully renamed to temporary names. Shuffling aborted.")
        return

    # 3. Shuffle the list of temporary file paths
    print("\n--- Phase 2: Shuffling temporary file paths ---")
    random.shuffle(temp_paths_after_first_pass)
    print("Temporary file paths have been randomly shuffled.")

    # 4. Second Pass: Rename from temporary names to new shuffled sequential names
    print("\n--- Phase 3: Renaming temporary files to final shuffled sequential names ---")

    num_digits = len(str(len(temp_paths_after_first_pass) - 1))
    if num_digits == 0:
        num_digits = 1

    for i, (temp_img_path, temp_lbl_path, img_ext) in enumerate(temp_paths_after_first_pass):
        new_base_name = f"frame_{i:0{num_digits}d}"

        final_img_path = os.path.join(IMAGES_PATH, f"{new_base_name}{img_ext}")
        final_lbl_path = os.path.join(LABEL_PATH, f"{new_base_name}.txt") if temp_lbl_path else None

        try:
            os.rename(temp_img_path, final_img_path)
            if temp_lbl_path:  # Only rename label if it existed as a temporary file
                os.rename(temp_lbl_path, final_lbl_path)
        except FileNotFoundError:
            print(
                f"Error: Temporary file not found during final rename for '{os.path.basename(temp_img_path)}' (or its label). This file might have been skipped earlier. Skipping.")
        except Exception as e:
            print(
                f"An unexpected error occurred during final rename of '{os.path.basename(temp_img_path)}': {e}. Skipping this pair.")

    print("\n--- Shuffling complete! ---")
    print(
        "All image files (and their corresponding labels if they exist) have been randomly shuffled and renamed sequentially.")


def split_and_shuffle_dataset(TRAIN_PATH, TRAIN_LABEL_PATH, VALIDATION_PATH, VALIDATION_LABEL_PATH):
    """
    Splits a YOLOv11 dataset into training (80%) and validation (20%) sets.
    It first shuffles the entire dataset (including images without labels)
    to ensure generalization and then moves the validation set images and their
    corresponding labels (if present) to separate directories.

    Args:
        TRAIN_PATH (str): The path to the directory containing the initial full training images.
        TRAIN_LABEL_PATH (str): The path to the directory containing the initial full training labels.
        VALIDATION_PATH (str): The path to the (initially empty) directory for validation images.
        VALIDATION_LABEL_PATH (str): The path to the (initially empty) directory for validation labels.
    """
    # 1. Validate paths
    if not os.path.isdir(TRAIN_PATH):
        print(f"Error: TRAIN_PATH '{TRAIN_PATH}' does not exist or is not a directory.")
        return
    if not os.path.isdir(TRAIN_LABEL_PATH):
        print(f"Error: TRAIN_LABEL_PATH '{TRAIN_LABEL_PATH}' does not exist or is not a directory.")
        return

    # Ensure validation directories exist (create if not)
    os.makedirs(VALIDATION_PATH, exist_ok=True)
    os.makedirs(VALIDATION_LABEL_PATH, exist_ok=True)

    # 2. Initial shuffle of the entire dataset within the TRAIN_PATH and TRAIN_LABEL_PATH
    # This randomizes the order of all frames and renames them sequentially (e.g., frame_000.jpg)
    print("\n--- Step 1: Shuffling the entire dataset in TRAIN_PATH and TRAIN_LABEL_PATH ---")
    shuffle_frames_randomly(TRAIN_PATH, TRAIN_LABEL_PATH)
    print("Initial dataset shuffling complete. Files are now sequentially named and randomized.")

    # 3. Re-gather files after shuffling and renaming to get the new, consistent names
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    # Get all image files with their NEW sequential names, applying the same filters as shuffle_frames_randomly
    all_image_files = []  # Will store image filenames (e.g., 'frame_000.jpg')
    for f in os.listdir(TRAIN_PATH):
        full_path = os.path.join(TRAIN_PATH, f)
        if os.path.isfile(full_path) and f.lower().endswith(image_extensions):
            if f.startswith('.'):
                continue  # Skip hidden files
            all_image_files.append(f)

    # Get all label files with their NEW sequential names
    # Note: This will only pick up labels for images that had labels before the shuffle.
    all_label_files_in_dir = [f for f in os.listdir(TRAIN_LABEL_PATH) if
                              os.path.isfile(os.path.join(TRAIN_LABEL_PATH, f)) and f.lower().endswith('.txt')]
    labels_dict = {os.path.splitext(f)[0]: f for f in all_label_files_in_dir}

    # Form pairs that include image filenames and their label filenames (or None if no label)
    # This list will be shuffled for the split
    all_files_for_split = []  # Stores (image_filename, label_filename_or_None)
    for img_file in all_image_files:
        base_name_without_ext = os.path.splitext(img_file)[0]
        corresponding_label_file = labels_dict.get(base_name_without_ext)
        all_files_for_split.append((img_file, corresponding_label_file))

    if not all_files_for_split:
        print("No image files found after initial shuffle. Cannot proceed with split.")
        return

    # 4. Shuffle the combined list to ensure random distribution for the split
    random.shuffle(all_files_for_split)

    print(f"\nFound {len(all_files_for_split)} image files (some without labels) for splitting.")

    # 5. Determine split points (80% train, 20% validation)
    num_total_frames = len(all_files_for_split)
    num_validation_frames = int(num_total_frames * 0.20)

    if num_validation_frames == 0 and num_total_frames > 0:
        num_validation_frames = 1
        print("Dataset size is very small. Ensuring at least 1 frame for validation.")
    elif num_total_frames == 0:
        print("No frames to split. Exiting.")
        return

    validation_set_items = all_files_for_split[:num_validation_frames]
    training_set_items_remaining = all_files_for_split[num_validation_frames:]

    print(
        f"Splitting dataset: {len(training_set_items_remaining)} for training, {len(validation_set_items)} for validation.")

    # 6. Move validation files
    print("\n--- Step 2: Moving validation files to VALIDATION_PATH and VALIDATION_LABEL_PATH ---")
    moved_count = 0
    for img_file, lbl_file in validation_set_items:
        current_img_path = os.path.join(TRAIN_PATH, img_file)
        dest_img_path = os.path.join(VALIDATION_PATH, img_file)

        try:
            shutil.move(current_img_path, dest_img_path)
            moved_count += 1

            if lbl_file:  # Only move label if it exists
                current_lbl_path = os.path.join(TRAIN_LABEL_PATH, lbl_file)
                dest_lbl_path = os.path.join(VALIDATION_LABEL_PATH, lbl_file)
                shutil.move(current_lbl_path, dest_lbl_path)
                # print(f"Moved '{img_file}' and '{lbl_file}' to validation.")
            else:
                # print(f"Moved '{img_file}' to validation (no associated label).")
                pass  # This is expected for unannotated images

        except FileNotFoundError:
            print(f"Error: File not found during move operation for '{img_file}' (or its label). Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while moving '{img_file}': {e}. Skipping.")

    print(f"\n--- Dataset split complete! Moved {moved_count} images to validation. ---")
    print(
        f"Training set now contains {len(os.listdir(TRAIN_PATH)) - len([f for f in os.listdir(TRAIN_PATH) if f.startswith('.') or os.path.isdir(os.path.join(TRAIN_PATH, f))])} valid images.")
    print(
        f"Validation set now contains {len(os.listdir(VALIDATION_PATH)) - len([f for f in os.listdir(VALIDATION_PATH) if f.startswith('.') or os.path.isdir(os.path.join(VALIDATION_PATH, f))])} valid images.")


# Function to process and convert images
def process_and_convert_images(input_img_dir, output_img_dir):
    os.makedirs(output_img_dir, exist_ok=True)

    # Assuming TIFFs are in `input_img_dir`
    image_files = [f for f in os.listdir(input_img_dir) if f.lower().endswith(('.tiff', '.tif'))]

    if not image_files:
        print(f"No TIFF files found in {input_img_dir}. Skipping conversion.")
        return

    for filename in tqdm(image_files, desc=f"Converting images in {os.path.basename(input_img_dir)}"):
        input_path = os.path.join(input_img_dir, filename)
        output_filename = os.path.splitext(filename)[0] + '.png'  # Change extension to .png
        output_path = os.path.join(output_img_dir, output_filename)

        try:
            # Use tifffile to load the original float32 CHW TIFF
            # tifffile.imread automatically handles float32 and CHW (if saved that way)
            img_chw_float32 = tifffile.imread(input_path)

            if img_chw_float32.ndim != 3 or img_chw_float32.shape[0] != 4:
                print(f"Warning: {input_path} has unexpected shape {img_chw_float32.shape}. Skipping.")
                continue

            # Assuming (0,1) float32 RGB and (0,1) float32 normalized depth
            # Channels are C, H, W (4, H, W) -> [R, G, B, Depth]

            # Extract channels
            rgb_float = img_chw_float32[:3, :, :]  # (3, H, W)
            depth_float = img_chw_float32[3, :, :]  # (H, W)

            # Scale RGB to 0-255 and convert to uint8 (still CHW)
            # Make sure to clip values to [0, 255] just in case of minor float errors
            rgb_uint8_chw = np.clip((rgb_float * 255.0), 0, 255).astype(np.uint8)

            # Scale Depth to 0-255 and convert to uint8 (still HW)
            # Again, clip to ensure valid uint8 range
            depth_uint8_hw = np.clip((depth_float * 255.0), 0, 255).astype(np.uint8)

            # Convert RGB from CHW to HWC for OpenCV
            rgb_uint8_hwc = rgb_uint8_chw.transpose((1, 2, 0))  # (H, W, 3)

            # Convert RGB (now HWC) to BGR for OpenCV's default saving (PNG is BGR+Alpha)
            rgb_uint8_bgr = cv2.cvtColor(rgb_uint8_hwc, cv2.COLOR_RGB2BGR)

            # Expand depth to HWC format (H, W, 1)
            depth_uint8_hwc = np.expand_dims(depth_uint8_hw, axis=2)

            # Concatenate to form a 4-channel HWC image (BGR + Depth) for cv2.imwrite
            final_4ch_img_uint8_hwc = np.concatenate((rgb_uint8_bgr, depth_uint8_hwc), axis=2)

            # Save as PNG. OpenCV treats 4-channel PNG as BGRA, which works for YOLO.
            cv2.imwrite(output_path, final_4ch_img_uint8_hwc)
            # print(f"Converted and saved: {output_path} (shape: {final_4ch_img_uint8_hwc.shape}, dtype: {final_4ch_img_uint8_hwc.dtype})")

        except Exception as e:
            print(f"Error processing {input_path}: {e}. Skipping.")











