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

# For the old two steps conversion
def convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset(input_base_dir, output_base_dir,input_image_subdir, output_image_subdir,input_val_subdir, output_val_subdir):
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)


    # Run conversion for train and val image directories
    process_and_convert_images(
        os.path.join(input_base_dir, input_image_subdir),
        os.path.join(output_base_dir, output_image_subdir)
    )
    process_and_convert_images(
        os.path.join(input_base_dir, input_val_subdir),
        os.path.join(output_base_dir, output_val_subdir)
    )

    # Assuming labels are in a 'labels' subdirectory parallel to 'images'
    input_label_dir_train = os.path.join(input_base_dir, 'labels/train')
    output_label_dir_train = os.path.join(output_base_dir, 'labels/train')
    os.makedirs(output_label_dir_train, exist_ok=True)
    for f in os.listdir(input_label_dir_train):
        if f.endswith('.txt'):
            os.link(os.path.join(input_label_dir_train, f), os.path.join(output_label_dir_train, f))
    print(f"Copied labels from {input_label_dir_train} to {output_label_dir_train}")

    input_label_dir_val = os.path.join(input_base_dir, 'labels/val')
    output_label_dir_val = os.path.join(output_base_dir, 'labels/val')
    os.makedirs(output_label_dir_val, exist_ok=True)
    for f in os.listdir(input_label_dir_val):
        if f.endswith('.txt'):
            os.link(os.path.join(input_label_dir_val, f), os.path.join(output_label_dir_val, f))
    print(f"Copied labels from {input_label_dir_val} to {output_label_dir_val}")

    print("Dataset conversion and label copying complete!")








