import cv2
import numpy as np
import os
import shutil
from PIL import Image # For saving multi-channel TIFF
from lib.utility import load_exr_depth, normalize_channel, generate_file_hash, copy_and_rename
from lib.global_dataset_utility import calculate_global_min_max

def fuse():
    # Directories
    home_dir = os.path.expanduser("~")
    downloads_dir = os.path.join(home_dir, "Downloads")

    # Input dataset
    RGB_FRAMES_DIR = downloads_dir + '/Frames'
    DEPTH_FRAMES_DIR = downloads_dir + "/Depth"
    ORIGINAL_LABELS_DIR = downloads_dir + '/Labels'

    # Output directories for the new multispectral dataset
    OUTPUT_DATASET_ROOT = downloads_dir + '/Dataset'
    OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DATASET_ROOT, 'images')
    OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DATASET_ROOT, 'labels')

    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    # List all RGB frames (assuming they are named sequentially, e.g., 00000.png)
    rgb_files = sorted([f for f in os.listdir(RGB_FRAMES_DIR) if f.endswith(('.png', '.jpg'))])

    # Calculate max and min depth values measured on the entire dataset
    print("Calculating global min/max depth for consistent normalization...")
    global_min_depth, global_max_depth = calculate_global_min_max(rgb_files, DEPTH_FRAMES_DIR)

    # Calculate global min based on a fixed max Laidar distance value
    # global_min_depth, global_max_depth = calculate_min_depth_with_fixed_max(rgb_files, DEPTH_FRAMES_DIR, LAIDAR_MAX_DEPTH)

    # --- Main Processing Loop ---
    print("\nProcessing frames and generating multi-channel TIFFs...")
    for i, rgb_filename in enumerate(rgb_files):
        base_filename = os.path.splitext(rgb_filename)[0]
        depth_filename = f"{base_filename}.exr"
        label_filename = f"{base_filename}.txt"  # Assuming label file has same base name

        rgb_path = os.path.join(RGB_FRAMES_DIR, rgb_filename)
        depth_path = os.path.join(DEPTH_FRAMES_DIR, depth_filename)
        label_path = os.path.join(ORIGINAL_LABELS_DIR, label_filename)

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path) or not os.path.exists(label_path):
            print(f"Skipping frame {i}: Missing RGB, Depth, or Label file.")
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

        # Ensure all inputs have the same resolution
        H, W, _ = rgb_frame.shape  # H, W from RGB image
        if depth_map.shape != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)
            print(f"Resized depth map for frame {i} to {W}x{H}")

        # Normalize depth (0-1 range)
        normalized_depth = normalize_channel(depth_map, min_val=global_min_depth, max_val=global_max_depth,
                                             target_range=(0, 1))

        # All arrays must be float32 for concatenation
        # Ensure RGB is also float32 and normalized 0-1
        rgb_float = rgb_frame.astype(np.float32) / 255.0

        # Expand dims for single-channel maps to (H, W, 1)
        normalized_depth = np.expand_dims(normalized_depth, axis=2)

        # Concatenate all channels along the last axis
        stacked_channels = np.concatenate((
            rgb_float,
            normalized_depth,
        ), axis=2)

        # Verify final shape and data type
        print(f"Frame {i}: Stacked channels shape: {stacked_channels.shape}, dtype: {stacked_channels.dtype}")

        # --- Save as Multi-Channel TIFF (Float32) ---
        output_tiff_path = os.path.join(OUTPUT_IMAGES_DIR, f"{base_filename}.tiff")

        # PIL expects image data as (H, W, C) or (H, W) for single channel.
        # For multi-channel float data, `Image.fromarray` is the way.
        # Specify the mode as 'F' (float) or 'F;16'/'F;32' for specific bit depths if needed.
        # For float32, mode='F' usually implies 32-bit float per pixel.
        # Pillow's `Image.fromarray` directly handles (H, W, C) numpy arrays correctly for TIFF.
        try:

            # Float32:
            pil_image = Image.fromarray(stacked_channels, mode='F')  # Pillow's 'F' mode is for float32

            # Save with appropriate TIFF tags if necessary
            pil_image.save(output_tiff_path, compression="tiff_deflate")  # tiff_deflate is a good lossless compression
            print(f"Saved multi-channel TIFF: {output_tiff_path}")
        except Exception as e:
            print(f"Error saving TIFF {output_tiff_path}: {e}")
            continue

        # --- Copy Labels to New Dataset Structure ---
        output_label_path = os.path.join(OUTPUT_LABELS_DIR, label_filename)
        # The label format is already "0 0.48 0.63 0.65 0.71" which is exactly what YOLO needs.

        shutil.copyfile(label_path, output_label_path)
        print(f"Copied label file: {output_label_path}")

    print("\nAll frames processed and saved as multi-channel TIFFs with corresponding labels.")


def find_correct_exr_and_fix_it(dataset_jpg_dir,array_of_jpg_and_exr_dirs,exr_output_dir):
    """array_of_jpg_and_exr_dirs is expected to be an array of directories each of which contains two subdirectories rgb for jpg images and depth for exr images"""

    jpg_dataset_hashes = {}
    jpg_no_dataset_hashes = {}

    print(f"Scanning dataset dir: {dataset_jpg_dir}...")
    for filename in os.listdir(dataset_jpg_dir):
        if filename.lower().endswith((".jpg", ".jpeg")):
            file_path = os.path.join(dataset_jpg_dir, filename)

            if os.path.isfile(file_path):
                file_hash = generate_file_hash(file_path)
                jpg_dataset_hashes[file_hash] = file_path
            else:
                print(f"Skipping file {file_path} as it is not a JPG.")


    for father_dir in array_of_jpg_and_exr_dirs:
        print(f"Scanning no dataset dir: {father_dir}...")
        single_no_dataset_jpg_dir = os.path.join(father_dir, "rgb")
        for filename in os.listdir(single_no_dataset_jpg_dir):
            if filename.lower().endswith((".jpg", ".jpeg")):
                single_exr_dir = os.path.join(father_dir, "depth")
                exr_file_path = os.path.join(single_exr_dir, filename)
                no_dataset_jpg_file_path = os.path.join(single_no_dataset_jpg_dir, filename)

                if os.path.isfile(no_dataset_jpg_file_path) and os.path.isfile(exr_file_path):
                    file_hash = generate_file_hash(no_dataset_jpg_file_path)
                    jpg_no_dataset_hashes[file_hash] = exr_file_path
                else:
                    print(f"Skipping file {no_dataset_jpg_file_path} and {exr_file_path} as one of them doesn't exist.")

    print('Comparing JPG files...')
    for hash_val, dataset_jpg_path in jpg_dataset_hashes.items():
        if hash_val in jpg_no_dataset_hashes:
            # We found a match! XD
            os.makedirs(exr_output_dir, exist_ok=True)
            new_name = os.path.basename(dataset_jpg_path)

            #destination_filepath = os.path.join(exr_output_dir,new_name)
            exr_source_filepath = jpg_no_dataset_hashes[hash_val]
            try:
                copy_and_rename(exr_source_filepath, exr_output_dir, new_name)
            except FileNotFoundError:
                print(f"Error: Source file not found at '{exr_source_filepath}'")
            except Exception as e:
                print(f"An error occurred during copying: {e}")






