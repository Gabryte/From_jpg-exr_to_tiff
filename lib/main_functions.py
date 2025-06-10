import cv2
import numpy as np
import os
import tifffile
from PIL import Image # For saving multi-channel TIFF
from lib.utility import load_exr_depth, normalize_channel, generate_file_hash, copy_and_rename_file
from lib.global_dataset_utility import calculate_global_min_max, down_grade_resolution_in_four_thirds


def fuse(RGB_FRAMES_DIR, DEPTH_FRAMES_DIR, OUTPUT_TIFF_DIR,TARGET_WIDTH):
    # Directories
    #home_dir = os.path.expanduser("~")
    #downloads_dir = os.path.join(home_dir, "Downloads")

    # Input dataset
    #RGB_FRAMES_DIR = downloads_dir + '/Frames'
    #DEPTH_FRAMES_DIR = downloads_dir + "/Depth"
    #ORIGINAL_LABELS_DIR = downloads_dir + '/Labels'

    # Output directories for the new multispectral dataset
    #OUTPUT_DATASET_ROOT = downloads_dir + '/Dataset'
    #OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DATASET_ROOT, 'images')
    #OUTPUT_LABELS_DIR = os.path.join(OUTPUT_DATASET_ROOT, 'labels')

    os.makedirs(OUTPUT_TIFF_DIR, exist_ok=True)


    # List all RGB frames (assuming they are named sequentially, e.g., 00000.jpg)
    rgb_files = sorted([f for f in os.listdir(RGB_FRAMES_DIR) if f.endswith(('.jpeg', '.jpg'))])

    # Calculate max and min depth values measured on the entire dataset
    print("Calculating global min/max depth for consistent normalization...")
    global_min_depth, global_max_depth = calculate_global_min_max(rgb_files, DEPTH_FRAMES_DIR)

    # Calculate global min based on a fixed max Laidar distance value
    # global_min_depth, global_max_depth = calculate_min_depth_with_fixed_max(rgb_files, DEPTH_FRAMES_DIR, LAIDAR_MAX_DEPTH)

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
            rgb_frame,depth_map = down_grade_resolution_in_four_thirds(TARGET_WIDTH=TARGET_WIDTH,rgb_frame=rgb_frame, depth_map=depth_map)


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
        except ImportError:
            # PIL expects image data as (H, W, C) or (H, W) for single channel.
            # For multi-channel float data, `Image.fromarray` is the way.
            # Specify the mode as 'F' (float) or 'F;16'/'F;32' for specific bit depths if needed.
            # For float32, mode='F' usually implies 32-bit float per pixel.
            # Pillow's `Image.fromarray` directly handles (H, W, C) numpy arrays correctly for TIFF.
            try:

                # Float32:
                pil_image = Image.fromarray(stacked_channels_hwc, mode='F')  # Pillow's 'F' mode is for float32

                # Save with appropriate TIFF tags if necessary
                pil_image.save(output_tiff_path, compression="tiff_deflate")  # tiff_deflate is a good lossless compression
                print(f"Saved multi-channel TIFF HWC: {output_tiff_path}")
            except Exception as e:
                print(f"Error saving TIFF {output_tiff_path}: {e}")
                continue

    print("\nAll frames processed and saved as multi-channel TIFFs")


def find_correct_exr_and_fix_it(dataset_jpg_dir,array_of_jpg_and_exr_dirs,exr_output_dir):
    """array_of_jpg_and_exr_dirs is expected to be an array of directories each of which contains two subdirectories rgb for jpg images and depth for exr images"""

    jpg_dataset_hashes = {}
    jpg_no_dataset_hashes = {}

    print(f"Scanning dataset dir: {dataset_jpg_dir}...")
    for filename in os.listdir(dataset_jpg_dir):
        if filename.lower().endswith((".jpg", ".jpeg")) and not filename.startswith("."):
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
            if filename.lower().endswith((".jpg", ".jpeg")) and not filename.startswith("."):
                single_exr_dir = os.path.join(father_dir, "depth")
                file_name_without_ext = os.path.splitext(filename)[0]
                exr_file_path = os.path.join(single_exr_dir, file_name_without_ext + ".exr")
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
            print(f"Match found between {hash_val} and {dataset_jpg_path}")
            os.makedirs(exr_output_dir, exist_ok=True)
            new_name = os.path.basename(dataset_jpg_path)

            new_true_base_name = os.path.splitext(new_name)[0]
            new_true_name = new_true_base_name + ".exr"


            #destination_filepath = os.path.join(exr_output_dir,new_name)
            exr_source_filepath = jpg_no_dataset_hashes[hash_val]
            old_name = os.path.basename(exr_source_filepath)
            exr_source_path = os.path.dirname(exr_source_filepath)
            if os.path.isfile(exr_source_filepath):
                print("Copying exr file...")
                try:
                    copy_and_rename_file(exr_source_path, exr_output_dir,old_name,new_true_name)
                except FileNotFoundError:
                    print(f"Error: Source file not found at '{exr_source_filepath}'")
                except Exception as e:
                    print(f"An error occurred during copying: {e}")
            else:
                print(f"Copy Failed: Source exr file not found at '{exr_source_filepath}'")







