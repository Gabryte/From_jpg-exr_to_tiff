import shutil

import cv2
import numpy as np
import os
from tqdm import tqdm
from lib.utility import normalize_array_to_range #,load_exr_depth, normalize_channel
from lib.exr_functions import load_single_channel_exr_map
#from lib.global_dataset_utility import calculate_global_min_max, downgrade_resolution_in_four_thirds_with_depths, \
#    process_and_convert_images
from lib.global_dataset_functions.resize import resize_rgb_and_depth_maintain_aspect_ratio


# --- Core Conversion Function: Processes and Fuses ALL data into a temp directory ---
def process_and_fuse_all_to_tiff(rgb_src_dir, depth_src_dir, labels_src_dir, temp_output_base_dir, global_min_log_depth,
                                global_max_log_depth, TARGET_WIDTH):
    """
    Processes all raw RGB images and corresponding EXR depth maps, fusing them
    into 4-channel TIFF files (RGB + normalized depth).
    Optionally, it also copies associated label files.

    The depth map is normalized using a logarithmic scale and global min/max
    values to ensure consistent scaling across the dataset.

    Args:
        rgb_src_dir (str): Path to the directory containing raw RGB images (e.g., .jpeg, .jpg, .png).
        depth_src_dir (str): Path to the directory containing raw depth maps (EXR format).
        labels_src_dir (str): Path to the directory containing label files (e.g., .txt).
                               If a label file is missing for an image, the image
                               will still be processed, but no label will be copied.
        temp_output_base_dir (str): Base directory where processed TIFF images and
                                    copied labels will be stored. Subdirectories
                                    'images/train' and 'labels/train' will be created.
        global_min_log_depth (float): The global minimum logarithmic depth value
                                      used for normalizing depth maps. This should
                                      be pre-calculated from the entire dataset.
        global_max_log_depth (float): The global maximum logarithmic depth value
                                      used for normalizing depth maps. This should
                                      be pre-calculated from the entire dataset.
        TARGET_WIDTH (int or None): If an integer, all processed images and depth
                                    maps will be resized to this width while
                                    maintaining their aspect ratio. If None,
                                    images will retain their original resolution.

    Returns:
        int: The number of successfully processed and fused frames.
    """
    # Define output directories for processed images and labels within the temporary base directory.
    # These paths are structured to support a typical 'train' dataset split.
    temp_images_dir = os.path.join(temp_output_base_dir, 'images', 'train')
    temp_labels_dir = os.path.join(temp_output_base_dir, 'labels','train')

    # Create the output directories if they don't already exist.
    # exist_ok=True prevents an error if the directories already exist.
    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    # Get a sorted list of RGB image filenames from the source directory.
    # Sorting ensures consistent processing order.
    rgb_files = sorted([f for f in os.listdir(rgb_src_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

    # Check if any RGB files were found. If not, print a message and exit.
    if not rgb_files:
        print(f"No RGB files found in {rgb_src_dir}. No images to process.")
        return 0 # Return 0 processed files.

    print(f"\nProcessing all raw frames and generating fused multi-channel TIFFs in '{temp_output_base_dir}'...")

    processed_count = 0 # Initialize a counter for successfully processed frames.

    # Iterate through each RGB file using tqdm for a progress bar.
    for i, rgb_filename in enumerate(tqdm(rgb_files, desc="Fusing and Converting")):
        # Extract the base filename (without extension) to find corresponding depth and label files.
        base_filename = os.path.splitext(rgb_filename)[0]
        depth_filename = f"{base_filename}.exr"  # EXR is common for depth maps due to high precision.
        label_filename = f"{base_filename}.txt"  # Assuming label files are plain text.

        # Construct full paths for the source RGB, depth, and label files.
        rgb_path = os.path.join(rgb_src_dir, rgb_filename)
        depth_path = os.path.join(depth_src_dir, depth_filename)
        label_path = os.path.join(labels_src_dir, label_filename)

        # Construct full paths for the output TIFF image and label file.
        # The output image will have a .tiff extension.
        output_tiff_path = os.path.join(temp_images_dir, f"{base_filename}.tiff")
        output_label_path = os.path.join(temp_labels_dir, label_filename)

        # --- Check for mandatory input files (RGB and Depth) ---
        # If an RGB file is missing, skip the current frame and log a message.
        if not os.path.exists(rgb_path):
            print(f"Skipping frame {base_filename}: Missing RGB file: {rgb_path}")
            continue # Move to the next iteration in the loop.
        # If a Depth file is missing, skip the current frame and log a message.
        if not os.path.exists(depth_path):
            print(f"Skipping frame {base_filename}: Missing Depth file: {depth_path}")
            continue # Move to the next iteration in the loop.

        # --- Check for optional input file (Label) ---
        # Determine if the label file exists. This flag will be used later.
        label_exists = os.path.exists(label_path)
        #if not label_exists:
            # Log a warning if the label file is not found, but continue processing the image.
            #print(f"Warning: Label file not found for {base_filename}. Image will be processed, but no label copied.")


        # Load RGB frame using OpenCV.
        # cv2.imread loads images as BGR by default.
        rgb_frame = cv2.imread(rgb_path)
        if rgb_frame is None:
            print(f"Error loading RGB frame {rgb_path}. Skipping.")
            continue
        # Convert RGB frame from BGR to RGB color space, as typically expected.
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Load Depth map using a custom helper function (load_single_channel_exr_map).
        # This function is expected to handle EXR specifics and return a single-channel float map.
        depth_map = load_single_channel_exr_map(depth_path)
        if depth_map is None:
            print(f"Error loading/processing depth for frame {depth_path}. Skipping.")
            continue

        # Resize RGB and Depth if TARGET_WIDTH is specified.
        # This maintains aspect ratio and resizes both images consistently.
        if TARGET_WIDTH:
            rgb_frame, depth_map = resize_rgb_and_depth_maintain_aspect_ratio(
                TARGET_WIDTH=TARGET_WIDTH, rgb_frame=rgb_frame, depth_map=depth_map
            )
        else:
            # If no target width, ensure depth map matches RGB dimensions in case of discrepancy.
            H, W, _ = rgb_frame.shape
            if depth_map.shape != (H, W):
                # Resize depth map to match RGB dimensions using linear interpolation.
                depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize depth map using a logarithmic transformation and global min/max values.
        # This makes the depth values more uniformly distributed and robust to outliers.
        log_depth_map = np.log1p(depth_map) # Applies log(1 + x) to handle zero or small depth values.
        normalized_depth_float = normalize_array_to_range(
            log_depth_map,
            min_val=global_min_log_depth,
            max_val=global_max_log_depth,
            target_range=(0, 1) # Normalize to a [0, 1] float range.
        )

        # Convert the normalized float depth to an 8-bit unsigned integer (0-255).
        # This is suitable for image formats that store pixel values in 8-bit.
        depth_uint8 = np.clip((normalized_depth_float * 255.0), 0, 255).astype(np.uint8)

        # Convert the RGB frame back to BGR for OpenCV's imwrite function.
        rgb_uint8_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        # Expand dimensions of the 8-bit depth map to (Height, Width, 1) to allow concatenation.
        depth_uint8_hwc = np.expand_dims(depth_uint8, axis=2)

        # Concatenate the BGR RGB channels with the single depth channel.
        # This creates a 4-channel image (B, G, R, Depth).
        final_4ch_img_uint8_hwc = np.concatenate((rgb_uint8_bgr, depth_uint8_hwc), axis=2)

        # Save the 4-channel image as a TIFF file. TIFF supports multi-channel images.
        cv2.imwrite(output_tiff_path, final_4ch_img_uint8_hwc)

        # Conditionally copy the label file if it exists.
        if label_exists:
            shutil.copy2(label_path, output_label_path) # copy2 preserves metadata.

        processed_count += 1 # Increment the counter for successfully processed frames.

    print(f"Finished fusing and converting {processed_count} frames to TIFFs.")
    return processed_count # Return the total count of processed frames.

# --- Old and not working for saving the best.pt in 4 channel format
# def process_and_fuse_all_to_png(rgb_src_dir, depth_src_dir, labels_src_dir, temp_output_base_dir, global_min_log_depth,
#                                 global_max_log_depth,TARGET_WIDTH):
#     """
#     Processes all raw RGB/EXR pairs, fuses them into 4-channel PNGs.
#     If a label file is missing, the image is still processed, but no label file is copied.
#     """
#     temp_images_dir = os.path.join(temp_output_base_dir, 'images', 'train')
#     temp_labels_dir = os.path.join(temp_output_base_dir, 'labels','train')
#
#     os.makedirs(temp_images_dir, exist_ok=True)
#     os.makedirs(temp_labels_dir, exist_ok=True)
#
#     rgb_files = sorted([f for f in os.listdir(rgb_src_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
#
#     if not rgb_files:
#         print(f"No RGB files found in {rgb_src_dir}. No images to process.")
#         return 0
#
#     print(f"\nProcessing all raw frames and generating fused multi-channel PNGs in '{temp_output_base_dir}'...")
#     processed_count = 0
#     for i, rgb_filename in enumerate(tqdm(rgb_files, desc="Fusing and Converting")):
#         base_filename = os.path.splitext(rgb_filename)[0]
#         depth_filename = f"{base_filename}.exr"
#         label_filename = f"{base_filename}.txt"
#
#         rgb_path = os.path.join(rgb_src_dir, rgb_filename)
#         depth_path = os.path.join(depth_src_dir, depth_filename)
#         label_path = os.path.join(labels_src_dir, label_filename)
#
#         output_png_path = os.path.join(temp_images_dir, f"{base_filename}.png")
#         output_label_path = os.path.join(temp_labels_dir, label_filename)
#
#         # --- Check for RGB and Depth (these are mandatory) ---
#         if not os.path.exists(rgb_path):
#             print(f"Skipping frame {base_filename}: Missing RGB file: {rgb_path}")
#             continue
#         if not os.path.exists(depth_path):
#             print(f"Skipping frame {base_filename}: Missing Depth file: {depth_path}")
#             continue
#
#         # --- Check for Label  ---
#         label_exists = os.path.exists(label_path)
#         #if not label_exists:
#             # Changed from 'continue' to just a print statement
#             #print(f"Warning: Label file not found for {base_filename}. Image will be processed, but no label copied.")
#
#
#         # Load RGB frame
#         rgb_frame = cv2.imread(rgb_path)
#         if rgb_frame is None:
#             print(f"Error loading RGB frame {rgb_path}. Skipping.")
#             continue
#         rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
#
#         # Load Depth map
#         depth_map = load_single_channel_exr_map(depth_path)
#         if depth_map is None:
#             print(f"Error loading/processing depth for frame {depth_path}. Skipping.")
#             continue
#
#         # Resize if TARGET_WIDTH is specified
#         if TARGET_WIDTH:
#             rgb_frame, depth_map = resize_rgb_and_depth_maintain_aspect_ratio(
#                 TARGET_WIDTH=TARGET_WIDTH, rgb_frame=rgb_frame, depth_map=depth_map
#             )
#         else:
#             H, W, _ = rgb_frame.shape
#             if depth_map.shape != (H, W):
#                 depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
#
#         # Normalize depth
#         log_depth_map = np.log1p(depth_map)
#         normalized_depth_float = normalize_array_to_range(
#             log_depth_map,
#             min_val=global_min_log_depth,
#             max_val=global_max_log_depth,
#             target_range=(0, 1)
#         )
#
#         depth_uint8 = np.clip((normalized_depth_float * 255.0), 0, 255).astype(np.uint8)
#         rgb_uint8_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
#         depth_uint8_hwc = np.expand_dims(depth_uint8, axis=2)
#
#         final_4ch_img_uint8_hwc = np.concatenate((rgb_uint8_bgr, depth_uint8_hwc), axis=2)
#
#         # Save as PNG
#         cv2.imwrite(output_png_path, final_4ch_img_uint8_hwc)
#
#         # --- Conditionally copy label file ---
#         if label_exists:  # Only copy if the label file was found
#             shutil.copy2(label_path, output_label_path)
#
#         processed_count += 1
#
#     print(f"Finished fusing and converting {processed_count} frames to PNGs.")
#     return processed_count

#old fuse function
# def fuse(RGB_FRAMES_DIR, DEPTH_FRAMES_DIR, OUTPUT_TIFF_DIR,TARGET_WIDTH):
#     """
#        RGB_FRAMES_DIR, DEPTH_FRAMES_DIR rgb frames that have a depth file linked must have the same name of the associated exr frame.
#         :param RGB_FRAMES_DIR: the input directory of RGB frames.
#         :param DEPTH_FRAMES_DIR: the input directory of depth frames.
#         :param OUTPUT_TIFF_DIR: the output directory in which the tifffile will be saved.
#         :param TARGET_WIDTH: the final with, it automatically resizes the images maintaining the aspect ratio
#         :return: you will have 4 channels tiff images in chw format with normalized depth and normalized rgb channels in [0-1] range
#     """
#     os.makedirs(OUTPUT_TIFF_DIR, exist_ok=True)
#
#
#     # List all RGB frames (assuming they are named sequentially, e.g., 00000.jpg)
#     rgb_files = sorted([f for f in os.listdir(RGB_FRAMES_DIR) if f.endswith(('.jpeg', '.jpg'))])
#
#     # Calculate max and min depth values measured on the entire dataset
#     print("Calculating global min/max depth for consistent normalization...")
#     global_min_depth, global_max_depth = calculate_global_min_max(rgb_files, DEPTH_FRAMES_DIR)
#
#     #global_min_depth, global_max_depth = 0.0302276611328125,11.25 for a specific train + test dataset
#
#     # --- Main Processing Loop ---
#     print("\nProcessing frames and generating multi-channel TIFFs in CHW format...")
#     for i, rgb_filename in enumerate(rgb_files):
#         base_filename = os.path.splitext(rgb_filename)[0]
#         depth_filename = f"{base_filename}.exr"
#
#
#         rgb_path = os.path.join(RGB_FRAMES_DIR, rgb_filename)
#         depth_path = os.path.join(DEPTH_FRAMES_DIR, depth_filename)
#
#
#         if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
#             print(f"Skipping frame {i}: Missing RGB or Depth")
#             continue
#
#         # Load RGB frame
#         rgb_frame = cv2.imread(rgb_path)
#         if rgb_frame is None:
#             print(f"Error loading RGB frame {rgb_path}. Skipping.")
#             continue
#         rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (OpenCV loads BGR)
#
#         # Load Depth frame
#         depth_map = load_exr_depth(depth_path)
#         if depth_map is None:
#             print(f"Error processing depth for frame {depth_path}. Skipping.")
#             continue
#
#
#         if TARGET_WIDTH:
#             rgb_frame,depth_map = downgrade_resolution_in_four_thirds_with_depths(TARGET_WIDTH=TARGET_WIDTH, rgb_frame=rgb_frame, depth_map=depth_map)
#
#
#         # Ensure all inputs have the same resolution
#         H, W, _ = rgb_frame.shape  # H, W from RGB image
#         if depth_map.shape != (H, W):
#             depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
#             print(f"Resized depth map for frame {i} to {W}x{H}")
#
#         # Normalize depth (0-1 range)
#         normalized_depth = normalize_channel(np.log1p(depth_map), min_val=global_min_depth, max_val=global_max_depth,
#                                              target_range=(0, 1))
#
#         # All arrays must be float32 for concatenation
#         # Ensure RGB is also float32 and normalized 0-1
#         rgb_float = rgb_frame.astype(np.float32) / 255.0
#
#         # Expand dims for single-channel maps to (H, W, 1)
#         normalized_depth = np.expand_dims(normalized_depth, axis=2)
#
#         # Concatenate all channels along the last axis
#         stacked_channels_hwc = np.concatenate((
#             rgb_float,
#             normalized_depth,
#         ), axis=2) #hwc
#
#         # --- Transpose to CHW before saving ---
#         stacked_channels_chw = stacked_channels_hwc.transpose((2, 0, 1))  # (C, H, W)
#
#         # Verify final shape and data type
#         print(f"Frame {i}: Stacked channels HWC shape: {stacked_channels_hwc.shape}, "
#               f"CHW shape: {stacked_channels_chw.shape}, dtype: {stacked_channels_chw.dtype}")
#
#
#         # --- Save as Multi-Channel TIFF (Float32) ---
#         output_tiff_path = os.path.join(OUTPUT_TIFF_DIR, f"{base_filename}.tiff")
#
#         try:
#             tifffile.imwrite(output_tiff_path, stacked_channels_chw, photometric='RGB', compression='deflate')
#             print(f"Saved multi-channel TIFF (CHW) using tifffile: {output_tiff_path}")
#         except ImportError as e:
#                 print(f"Error saving TIFF {output_tiff_path}: {e}")
#                 continue
#
#     print("\nAll frames processed and saved as multi-channel TIFFs")


# For the old two steps conversion
# def convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset(input_base_dir, output_base_dir,input_image_subdir, output_image_subdir,input_val_subdir, output_val_subdir,labels_type):
#     """
#         examples: 1. labels_type = 'labels/train' 2. labels_type = 'labels/Test'
#     """
#     # Create output base directory
#     os.makedirs(output_base_dir, exist_ok=True)
#
#
#     # Run conversion for type (train or test) and optional val image directories
#     process_and_convert_images(
#         os.path.join(input_base_dir, input_image_subdir),
#         os.path.join(output_base_dir, output_image_subdir)
#     )
#     val_sentinel = False
#     if input_val_subdir is not None:
#         val_sentinel = True
#         process_and_convert_images(
#             os.path.join(input_base_dir, input_val_subdir),
#             os.path.join(output_base_dir, output_val_subdir)
#         )
#
#     # Assuming labels are in a 'labels' subdirectory parallel to 'images'
#     input_label_dir_train = os.path.join(input_base_dir, labels_type)
#     output_label_dir_train = os.path.join(output_base_dir, labels_type)
#
#     if input_label_dir_train != output_label_dir_train:
#         os.makedirs(output_label_dir_train, exist_ok=True)
#         for f in os.listdir(input_label_dir_train):
#             if f.endswith('.txt'):
#                 os.link(os.path.join(input_label_dir_train, f), os.path.join(output_label_dir_train, f))
#         print(f"Copied labels from {input_label_dir_train} to {output_label_dir_train}")
#
#     if val_sentinel:
#         input_label_dir_val = os.path.join(input_base_dir, 'labels/val')
#         output_label_dir_val = os.path.join(output_base_dir, 'labels/val')
#         os.makedirs(output_label_dir_val, exist_ok=True)
#         for f in os.listdir(input_label_dir_val):
#             if f.endswith('.txt'):
#                 os.link(os.path.join(input_label_dir_val, f), os.path.join(output_label_dir_val, f))
#         print(f"Copied labels from {input_label_dir_val} to {output_label_dir_val}")
#
#     print("Dataset conversion and label copying complete!")








