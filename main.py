from lib.global_dataset_utility import calculate_global_min_max, split_and_shuffle_dataset, process_and_convert_images
from lib.main_functions import find_correct_exr_and_fix_it
from lib.utility import get_input_directories, visualize_multichannel_tiff
from lib.utility import find_missing_exr_for_jpg
from lib.main_functions import fuse
from lib.global_dataset_utility import resize_resolution_maintaining_aspect_ratio
import OpenEXR
import Imath
import numpy as np
import os
if __name__ == '__main__':
    #array_of_exr_and_jpg_dirs = get_input_directories()

    #find_correct_exr_and_fix_it("/home/jacobo/Downloads/mines_dataset/images/train",array_of_exr_and_jpg_dirs,'/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files')

    #missing = find_missing_exr_for_jpg("/home/jacobo/Downloads/mines_dataset/images/train",'/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files')

    #fuse('/home/jacobo/Downloads/mines_dataset/images/train','/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files','/home/jacobo/Downloads/mines_dataset/images/tiff',480)
    # List all RGB frames (assuming they are named sequentially, e.g., 00000.jpg)
    #rgb_files = sorted([f for f in os.listdir('/home/jacobo/Downloads/mines_dataset/images/train') if f.endswith(('.jpeg', '.jpg'))])

    # Calculate max and min depth values measured on the entire dataset
    #print("Calculating global min/max depth for consistent normalization...")
    #global_min_depth, global_max_depth = calculate_global_min_max(rgb_files, '/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files')
    #print("Global min depth: ", global_min_depth)
    #print("Global max depth: ", global_max_depth)

    #resize_resolution_maintaining_aspect_ratio(480,'/home/jacobo/Downloads/mines_dataset/images/train')

    #split_and_shuffle_dataset('/home/jacobo/Downloads/mines_multichannel_dataset/images/train','/home/jacobo/Downloads/mines_multichannel_dataset/labels/train','/home/jacobo/Downloads/mines_multichannel_dataset/images/val','/home/jacobo/Downloads/mines_multichannel_dataset/labels/val')

    # Define your main TIFF output directory
    #tiff_data_directory = "/home/jacobo/Downloads/tiff"

    # Define the directory where you want to save the preview images
    #preview_output_directory = "/home/jacobo/Downloads/tiff_previews"

    # List all TIFF files in your data directory
    #tiff_files_in_dir = sorted([f for f in os.listdir(tiff_data_directory) if f.endswith('.tiff')])

    #if tiff_files_in_dir:
        # Select the first TIFF file found for visualization
        #first_tiff_path = os.path.join(tiff_data_directory, tiff_files_in_dir[0])

        #print(f"Calling visualization for: {first_tiff_path}")
        # Call the refined function with the specific TIFF path and the preview output directory
        #visualize_multichannel_tiff(first_tiff_path, preview_output_directory)
    #else:
        #print(f"No TIFF files found in '{tiff_data_directory}'. Please ensure the processing has run and files exist.")



    # --- Configuration for Conversion ---
    input_base_dir = '/home/jacobo/dataset/mines_multichannel_dataset/'
    output_base_dir = '/home/jacobo/dataset/mines_multichannel_dataset_converted_png/'

    input_image_subdir = 'images/train'
    output_image_subdir = 'images/train'  # Keeping same subdirectory structure

    input_val_subdir = 'images/val'
    output_val_subdir = 'images/val'

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




