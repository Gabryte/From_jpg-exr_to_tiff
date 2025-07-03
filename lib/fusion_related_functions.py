import shutil

import cv2
import numpy as np
import os
import tifffile
from tqdm import tqdm
from lib.utility import load_exr_depth, normalize_channel, generate_file_hash, copy_and_rename_file, \
    normalize_array_to_range, load_single_channel_exr_map
from lib.global_dataset_utility import calculate_global_min_max, downgrade_resolution_in_four_thirds_with_depths, \
    process_and_convert_images, resize_rgb_and_depth_maintain_aspect_ratio


#new function with uint8 for depth that saves in tiff
def process_and_fuse_all_to_tiff(rgb_src_dir, depth_src_dir, labels_src_dir, temp_output_base_dir, global_min_log_depth,
                                global_max_log_depth, TARGET_WIDTH):
    """
    Processes all raw RGB/EXR pairs, fuses them into 4-channel TIFFs.
    If a label file is missing, the image is still processed, but no label file is copied.
    """
    temp_images_dir = os.path.join(temp_output_base_dir, 'images', 'train')
    temp_labels_dir = os.path.join(temp_output_base_dir, 'labels','train')

    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    rgb_files = sorted([f for f in os.listdir(rgb_src_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

    if not rgb_files:
        print(f"No RGB files found in {rgb_src_dir}. No images to process.")
        return 0

    print(f"\nProcessing all raw frames and generating fused multi-channel TIFFs in '{temp_output_base_dir}'...")
    processed_count = 0
    for i, rgb_filename in enumerate(tqdm(rgb_files, desc="Fusing and Converting")):
        base_filename = os.path.splitext(rgb_filename)[0]
        depth_filename = f"{base_filename}.exr"
        label_filename = f"{base_filename}.txt"

        rgb_path = os.path.join(rgb_src_dir, rgb_filename)
        depth_path = os.path.join(depth_src_dir, depth_filename)
        label_path = os.path.join(labels_src_dir, label_filename)

        # --- CHANGE 1: Output file extension changed to .tiff ---
        output_tiff_path = os.path.join(temp_images_dir, f"{base_filename}.tiff")
        output_label_path = os.path.join(temp_labels_dir, label_filename)

        # --- Check for RGB and Depth (these are mandatory) ---
        if not os.path.exists(rgb_path):
            print(f"Skipping frame {base_filename}: Missing RGB file: {rgb_path}")
            continue
        if not os.path.exists(depth_path):
            print(f"Skipping frame {base_filename}: Missing Depth file: {depth_path}")
            continue

        # --- Check for Label (optional, based on your request) ---
        label_exists = os.path.exists(label_path)
        if not label_exists:
            print(f"Warning: Label file not found for {base_filename}. Image will be processed, but no label copied.")

        # Load RGB frame
        rgb_frame = cv2.imread(rgb_path)
        if rgb_frame is None:
            print(f"Error loading RGB frame {rgb_path}. Skipping.")
            continue
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Load Depth map
        depth_map = load_single_channel_exr_map(depth_path)
        if depth_map is None:
            print(f"Error loading/processing depth for frame {depth_path}. Skipping.")
            continue

        # Resize if TARGET_WIDTH is specified
        if TARGET_WIDTH:
            rgb_frame, depth_map = resize_rgb_and_depth_maintain_aspect_ratio(
                TARGET_WIDTH=TARGET_WIDTH, rgb_frame=rgb_frame, depth_map=depth_map
            )
        else:
            H, W, _ = rgb_frame.shape
            if depth_map.shape != (H, W):
                depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize depth
        log_depth_map = np.log1p(depth_map)
        normalized_depth_float = normalize_array_to_range(
            log_depth_map,
            min_val=global_min_log_depth,
            max_val=global_max_log_depth,
            target_range=(0, 1)
        )

        depth_uint8 = np.clip((normalized_depth_float * 255.0), 0, 255).astype(np.uint8)
        rgb_uint8_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        depth_uint8_hwc = np.expand_dims(depth_uint8, axis=2)

        final_4ch_img_uint8_hwc = np.concatenate((rgb_uint8_bgr, depth_uint8_hwc), axis=2)

        # --- CHANGE 2: Save as TIFF using the new path ---
        cv2.imwrite(output_tiff_path, final_4ch_img_uint8_hwc)

        # --- Conditionally copy label file ---
        if label_exists:
            shutil.copy2(label_path, output_label_path)

        processed_count += 1

    print(f"Finished fusing and converting {processed_count} frames to TIFFs.")
    return processed_count

# --- Core Conversion Function: Processes and Fuses ALL data into a temp directory ---
def process_and_fuse_all_to_png(rgb_src_dir, depth_src_dir, labels_src_dir, temp_output_base_dir, global_min_log_depth,
                                global_max_log_depth,TARGET_WIDTH):
    """
    Processes all raw RGB/EXR pairs, fuses them into 4-channel PNGs.
    If a label file is missing, the image is still processed, but no label file is copied.
    """
    temp_images_dir = os.path.join(temp_output_base_dir, 'images', 'train')
    temp_labels_dir = os.path.join(temp_output_base_dir, 'labels','train')

    os.makedirs(temp_images_dir, exist_ok=True)
    os.makedirs(temp_labels_dir, exist_ok=True)

    rgb_files = sorted([f for f in os.listdir(rgb_src_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

    if not rgb_files:
        print(f"No RGB files found in {rgb_src_dir}. No images to process.")
        return 0

    print(f"\nProcessing all raw frames and generating fused multi-channel PNGs in '{temp_output_base_dir}'...")
    processed_count = 0
    for i, rgb_filename in enumerate(tqdm(rgb_files, desc="Fusing and Converting")):
        base_filename = os.path.splitext(rgb_filename)[0]
        depth_filename = f"{base_filename}.exr"
        label_filename = f"{base_filename}.txt"

        rgb_path = os.path.join(rgb_src_dir, rgb_filename)
        depth_path = os.path.join(depth_src_dir, depth_filename)
        label_path = os.path.join(labels_src_dir, label_filename)

        output_png_path = os.path.join(temp_images_dir, f"{base_filename}.png")
        output_label_path = os.path.join(temp_labels_dir, label_filename)

        # --- Check for RGB and Depth (these are mandatory) ---
        if not os.path.exists(rgb_path):
            print(f"Skipping frame {base_filename}: Missing RGB file: {rgb_path}")
            continue
        if not os.path.exists(depth_path):
            print(f"Skipping frame {base_filename}: Missing Depth file: {depth_path}")
            continue

        # --- Check for Label (optional, based on your request) ---
        label_exists = os.path.exists(label_path)
        if not label_exists:
            # Changed from 'continue' to just a print statement
            print(f"Warning: Label file not found for {base_filename}. Image will be processed, but no label copied.")

        # Load RGB frame
        rgb_frame = cv2.imread(rgb_path)
        if rgb_frame is None:
            print(f"Error loading RGB frame {rgb_path}. Skipping.")
            continue
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Load Depth map
        depth_map = load_single_channel_exr_map(depth_path)
        if depth_map is None:
            print(f"Error loading/processing depth for frame {depth_path}. Skipping.")
            continue

        # Resize if TARGET_WIDTH is specified
        if TARGET_WIDTH:
            rgb_frame, depth_map = resize_rgb_and_depth_maintain_aspect_ratio(
                TARGET_WIDTH=TARGET_WIDTH, rgb_frame=rgb_frame, depth_map=depth_map
            )
        else:
            H, W, _ = rgb_frame.shape
            if depth_map.shape != (H, W):
                depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Normalize depth
        log_depth_map = np.log1p(depth_map)
        normalized_depth_float = normalize_array_to_range(
            log_depth_map,
            min_val=global_min_log_depth,
            max_val=global_max_log_depth,
            target_range=(0, 1)
        )

        depth_uint8 = np.clip((normalized_depth_float * 255.0), 0, 255).astype(np.uint8)
        rgb_uint8_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        depth_uint8_hwc = np.expand_dims(depth_uint8, axis=2)

        final_4ch_img_uint8_hwc = np.concatenate((rgb_uint8_bgr, depth_uint8_hwc), axis=2)

        # Save as PNG
        cv2.imwrite(output_png_path, final_4ch_img_uint8_hwc)

        # --- Conditionally copy label file ---
        if label_exists:  # Only copy if the label file was found
            shutil.copy2(label_path, output_label_path)

        processed_count += 1

    print(f"Finished fusing and converting {processed_count} frames to PNGs.")
    return processed_count

#old fuse function
def fuse(RGB_FRAMES_DIR, DEPTH_FRAMES_DIR, OUTPUT_TIFF_DIR,TARGET_WIDTH):
    """
       RGB_FRAMES_DIR, DEPTH_FRAMES_DIR rgb frames that have a depth file linked must have the same name of the associated exr frame.
        :param RGB_FRAMES_DIR: the input directory of RGB frames.
        :param DEPTH_FRAMES_DIR: the input directory of depth frames.
        :param OUTPUT_TIFF_DIR: the output directory in which the tifffile will be saved.
        :param TARGET_WIDTH: the final with, it automatically resizes the images maintaining the aspect ratio
        :return: you will have 4 channels tiff images in chw format with normalized depth and normalized rgb channels in [0-1] range
    """
    os.makedirs(OUTPUT_TIFF_DIR, exist_ok=True)


    # List all RGB frames (assuming they are named sequentially, e.g., 00000.jpg)
    rgb_files = sorted([f for f in os.listdir(RGB_FRAMES_DIR) if f.endswith(('.jpeg', '.jpg'))])

    # Calculate max and min depth values measured on the entire dataset
    print("Calculating global min/max depth for consistent normalization...")
    global_min_depth, global_max_depth = calculate_global_min_max(rgb_files, DEPTH_FRAMES_DIR)

    #global_min_depth, global_max_depth = 0.0302276611328125,11.25 for a specific train + test dataset

    # --- Main Processing Loop ---
    print("\nProcessing frames and generating multi-channel TIFFs in CHW format...")
    for i, rgb_filename in enumerate(rgb_files):
        base_filename = os.path.splitext(rgb_filename)[0]
        depth_filename = f"{base_filename}.exr"


        rgb_path = os.path.join(RGB_FRAMES_DIR, rgb_filename)
        depth_path = os.path.join(DEPTH_FRAMES_DIR, depth_filename)


        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"Skipping frame {i}: Missing RGB or Depth")
            continue

        # Load RGB frame
        rgb_frame = cv2.imread(rgb_path)
        if rgb_frame is None:
            print(f"Error loading RGB frame {rgb_path}. Skipping.")
            continue
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (OpenCV loads BGR)

        # Load Depth frame
        depth_map = load_exr_depth(depth_path)
        if depth_map is None:
            print(f"Error processing depth for frame {depth_path}. Skipping.")
            continue


        if TARGET_WIDTH:
            rgb_frame,depth_map = downgrade_resolution_in_four_thirds_with_depths(TARGET_WIDTH=TARGET_WIDTH, rgb_frame=rgb_frame, depth_map=depth_map)


        # Ensure all inputs have the same resolution
        H, W, _ = rgb_frame.shape  # H, W from RGB image
        if depth_map.shape != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
            print(f"Resized depth map for frame {i} to {W}x{H}")

        # Normalize depth (0-1 range)
        normalized_depth = normalize_channel(np.log1p(depth_map), min_val=global_min_depth, max_val=global_max_depth,
                                             target_range=(0, 1))

        # All arrays must be float32 for concatenation
        # Ensure RGB is also float32 and normalized 0-1
        rgb_float = rgb_frame.astype(np.float32) / 255.0

        # Expand dims for single-channel maps to (H, W, 1)
        normalized_depth = np.expand_dims(normalized_depth, axis=2)

        # Concatenate all channels along the last axis
        stacked_channels_hwc = np.concatenate((
            rgb_float,
            normalized_depth,
        ), axis=2) #hwc

        # --- Transpose to CHW before saving ---
        stacked_channels_chw = stacked_channels_hwc.transpose((2, 0, 1))  # (C, H, W)

        # Verify final shape and data type
        print(f"Frame {i}: Stacked channels HWC shape: {stacked_channels_hwc.shape}, "
              f"CHW shape: {stacked_channels_chw.shape}, dtype: {stacked_channels_chw.dtype}")


        # --- Save as Multi-Channel TIFF (Float32) ---
        output_tiff_path = os.path.join(OUTPUT_TIFF_DIR, f"{base_filename}.tiff")

        try:
            tifffile.imwrite(output_tiff_path, stacked_channels_chw, photometric='RGB', compression='deflate')
            print(f"Saved multi-channel TIFF (CHW) using tifffile: {output_tiff_path}")
        except ImportError as e:
                print(f"Error saving TIFF {output_tiff_path}: {e}")
                continue

    print("\nAll frames processed and saved as multi-channel TIFFs")


def find_correct_exr_and_fix_it(dataset_jpg_dir, array_of_jpg_and_exr_dirs, exr_output_dir):
    """
    Identifies and copies the correct EXR depth maps based on matching JPG images' content.

    This function compares JPG images from a primary 'dataset' directory with JPG images
    found within various 'no_dataset' directories. When a JPG image in the 'dataset'
    directory has an identical content hash to a JPG image in a 'no_dataset' directory,
    its corresponding EXR depth map (found in the 'depth' subdirectory of the
    'no_dataset' structure) is copied to a specified output directory. The copied
    EXR file is renamed to match the base name of the 'dataset' JPG image.

    Args:
        dataset_jpg_dir (str): The path to the directory containing the "ground truth"
                               or reference JPG images. These images are hashed to find
                               matching counterparts.
        array_of_jpg_and_exr_dirs (list): A list of directory paths. Each directory in
                                         this list is expected to contain two specific
                                         subdirectories:
                                         - 'rgb/': For JPG images that potentially match
                                                   the 'dataset_jpg_dir' images.
                                         - 'depth/': For EXR depth map images that
                                                     correspond to the JPGs in 'rgb/'.
        exr_output_dir (str): The path to the directory where the correctly identified
                              and renamed EXR files will be copied. This directory will
                              be created if it doesn't exist.
    """

    # Dictionary to store hashes of JPG images from the dataset directory.
    # Key: File hash (str), Value: Full path to the JPG file (str).
    jpg_dataset_hashes = {}
    # Dictionary to store hashes of JPG images from the 'no_dataset' directories
    # and their corresponding EXR file paths.
    # Key: JPG file hash (str), Value: Full path to the associated EXR file (str).
    jpg_no_dataset_hashes = {}

    print(f"Scanning dataset directory: {dataset_jpg_dir} for JPG images and their hashes...")
    # Iterate through all files in the 'dataset_jpg_dir'.
    for filename in os.listdir(dataset_jpg_dir):
        # Check if the file is a JPG/JPEG and not a hidden file.
        if filename.lower().endswith((".jpg", ".jpeg")) and not filename.startswith("."):
            file_path = os.path.join(dataset_jpg_dir, filename)

            # Ensure the path points to an actual file before processing.
            if os.path.isfile(file_path):
                # Generate a hash for the JPG file's content.
                file_hash = generate_file_hash(file_path)
                # Store the hash and the file path in the dataset hashes dictionary.
                jpg_dataset_hashes[file_hash] = file_path
            else:
                # This case might occur if os.listdir returns directories or broken symlinks.
                print(f"Skipping '{file_path}' as it is not a valid JPG file or does not exist.")


    print(f"Scanning source directories for JPG-EXR pairs: {array_of_jpg_and_exr_dirs}...")
    # Iterate through each parent directory provided in `array_of_jpg_and_exr_dirs`.
    # Each 'father_dir' is expected to contain 'rgb' and 'depth' subdirectories.
    for father_dir in array_of_jpg_and_exr_dirs:
        print(f"Processing source directory: {father_dir}...")
        # Construct the path to the 'rgb' subdirectory within the current father_dir.
        single_no_dataset_jpg_dir = os.path.join(father_dir, "rgb")

        # Iterate through the JPG files within the 'rgb' subdirectory.
        for filename in os.listdir(single_no_dataset_jpg_dir):
            # Check if the file is a JPG/JPEG and not a hidden file.
            if filename.lower().endswith((".jpg", ".jpeg")) and not filename.startswith("."):
                # Construct the path to the 'depth' subdirectory within the current father_dir.
                single_exr_dir = os.path.join(father_dir, "depth")
                # Extract the base name of the JPG file (without extension).
                file_name_without_ext = os.path.splitext(filename)[0]
                # Construct the full path to the corresponding EXR file.
                exr_file_path = os.path.join(single_exr_dir, file_name_without_ext + ".exr")
                # Construct the full path to the current JPG file from the 'no_dataset' source.
                no_dataset_jpg_file_path = os.path.join(single_no_dataset_jpg_dir, filename)

                # Verify that both the JPG and its corresponding EXR file exist before processing.
                if os.path.isfile(no_dataset_jpg_file_path) and os.path.isfile(exr_file_path):
                    # Generate a hash for the content of the 'no_dataset' JPG file.
                    file_hash = generate_file_hash(no_dataset_jpg_file_path)
                    # Store the JPG hash and the path to its associated EXR file.
                    jpg_no_dataset_hashes[file_hash] = exr_file_path
                else:
                    print(f"Skipping processing for JPG '{no_dataset_jpg_file_path}' and EXR '{exr_file_path}' "
                          f"as one or both files do not exist.")

    print('Starting comparison of JPG files to find matching EXR depth maps...')
    # Now, compare the hashes from the dataset JPGs with the hashes from the 'no_dataset' JPGs.
    # This loop iterates through each JPG hash found in the primary 'dataset_jpg_dir'.
    for hash_val, dataset_jpg_path in jpg_dataset_hashes.items():
        # If the hash of a dataset JPG is found in the 'no_dataset' JPG hashes,
        # it means we've identified a content-identical JPG.
        if hash_val in jpg_no_dataset_hashes:
            print(f"Match found! Dataset JPG: '{dataset_jpg_path}' (Hash: {hash_val[:8]}...) "
                  f"matches a source JPG with corresponding EXR: '{jpg_no_dataset_hashes[hash_val]}'")

            # Ensure the output directory for EXR files exists. If not, create it.
            os.makedirs(exr_output_dir, exist_ok=True)

            # Get the base filename of the matched JPG from the dataset.
            # This name will be used to rename the copied EXR file.
            new_name = os.path.basename(dataset_jpg_path)

            # Extract the filename without its extension (e.g., "image123" from "image123.jpg").
            new_true_base_name = os.path.splitext(new_name)[0]
            # Construct the new filename for the EXR file (e.g., "image123.exr").
            new_true_name = new_true_base_name + ".exr"

            # Retrieve the full path to the EXR file corresponding to the matched JPG hash.
            exr_source_filepath = jpg_no_dataset_hashes[hash_val]
            # Get the original filename of the EXR file.
            old_name = os.path.basename(exr_source_filepath)
            # Get the directory of the original EXR file.
            exr_source_path = os.path.dirname(exr_source_filepath)

            # Perform the copy operation only if the source EXR file actually exists.
            if os.path.isfile(exr_source_filepath):
                print(f"Attempting to copy EXR file from '{exr_source_filepath}' to '{exr_output_dir}' as '{new_true_name}'...")
                try:
                    # Call the helper function to copy and rename the EXR file.
                    copy_and_rename_file(exr_source_path, exr_output_dir, old_name, new_true_name)
                except FileNotFoundError:
                    # Handle cases where the source file might unexpectedly not be found during the copy attempt.
                    print(f"Error: Source EXR file not found at '{exr_source_filepath}' during copy operation.")
                except Exception as e:
                    # Catch any other potential errors during the file copying process.
                    print(f"An unexpected error occurred during EXR file copying: {e}")
            else:
                # This should ideally not happen if the previous os.path.isfile check passed,
                # but it serves as an additional safeguard.
                print(f"Copy Failed: Source EXR file '{exr_source_filepath}' unexpectedly not found.")
        else:
            print(f"No matching EXR found for dataset JPG: '{dataset_jpg_path}' (Hash: {hash_val[:8]}...)")

    print("EXR finding and fixing process completed.")

# For the old two steps conversion
def convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset(input_base_dir, output_base_dir,input_image_subdir, output_image_subdir,input_val_subdir, output_val_subdir,labels_type):
    """
        examples: 1. labels_type = 'labels/train' 2. labels_type = 'labels/Test'
    """
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)


    # Run conversion for type (train or test) and optional val image directories
    process_and_convert_images(
        os.path.join(input_base_dir, input_image_subdir),
        os.path.join(output_base_dir, output_image_subdir)
    )
    val_sentinel = False
    if input_val_subdir is not None:
        val_sentinel = True
        process_and_convert_images(
            os.path.join(input_base_dir, input_val_subdir),
            os.path.join(output_base_dir, output_val_subdir)
        )

    # Assuming labels are in a 'labels' subdirectory parallel to 'images'
    input_label_dir_train = os.path.join(input_base_dir, labels_type)
    output_label_dir_train = os.path.join(output_base_dir, labels_type)

    if input_label_dir_train != output_label_dir_train:
        os.makedirs(output_label_dir_train, exist_ok=True)
        for f in os.listdir(input_label_dir_train):
            if f.endswith('.txt'):
                os.link(os.path.join(input_label_dir_train, f), os.path.join(output_label_dir_train, f))
        print(f"Copied labels from {input_label_dir_train} to {output_label_dir_train}")

    if val_sentinel:
        input_label_dir_val = os.path.join(input_base_dir, 'labels/val')
        output_label_dir_val = os.path.join(output_base_dir, 'labels/val')
        os.makedirs(output_label_dir_val, exist_ok=True)
        for f in os.listdir(input_label_dir_val):
            if f.endswith('.txt'):
                os.link(os.path.join(input_label_dir_val, f), os.path.join(output_label_dir_val, f))
        print(f"Copied labels from {input_label_dir_val} to {output_label_dir_val}")

    print("Dataset conversion and label copying complete!")








