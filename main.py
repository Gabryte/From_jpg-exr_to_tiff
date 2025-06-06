import cv2
import numpy as np
import os
import shutil
from PIL import Image # For saving multi-channel TIFF


from lib.utility import load_exr_depth, calculate_global_min_max, normalize_channel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Directories
    home_dir = os.path.expanduser("~")
    downloads_dir = os.path.join(home_dir, "Downloads")

    #Input dataset
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

    global_min_depth, global_max_depth = calculate_global_min_max(rgb_files, DEPTH_FRAMES_DIR)

    print("Calculating global min/max depth and height for consistent normalization...")
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
        normalized_depth = normalize_channel(depth_map,min_val=global_min_depth, max_val=global_max_depth,target_range=(0, 1))

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