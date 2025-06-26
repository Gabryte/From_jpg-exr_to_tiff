from lib.global_dataset_utility import calculate_global_min_max, split_and_shuffle_dataset, process_and_convert_images, \
    calculate_log_depth_global_min_max
from lib.fusion_related_functions import find_correct_exr_and_fix_it, \
    convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset, process_and_fuse_all_to_png
from lib.utility import get_input_directories, visualize_multichannel_tiff
from lib.utility import find_missing_exr_for_jpg
from lib.fusion_related_functions import fuse
from lib.global_dataset_utility import resize_resolution_maintaining_aspect_ratio
import OpenEXR
import Imath
import numpy as np
import os
if __name__ == '__main__':
    #Getting all the different directories of the exported images in jpg and exr files, because a single directory contains the extracted frames of a single video from the record 3d app of the iphone 16
    #array_of_exr_and_jpg_dirs = get_input_directories()

    # --- Configuration needed in order to create a single training rgb dataset that have all of it's corresponding exr files associated via the same name. It's important for the next step of fusion between rgb and exr frames ---
    #find_correct_exr_and_fix_it("/home/jacobo/Downloads/mines_dataset/images/train",array_of_exr_and_jpg_dirs,'/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files')

    # --- Configuration in order to find eventually missing exr files for rgb images due to iphone 16 export errors ---
    #missing = find_missing_exr_for_jpg("/home/jacobo/Downloads/mines_dataset/images/train",'/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files')

    # --- Configuration in order to create a 4 channel tiff dataset, the result dataset needs to be converted into a png dataset compatible with yolo11 using convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset function---
    #fuse('/home/jacobo/Downloads/mines_dataset/images/train','/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files','/home/jacobo/Downloads/mines_dataset/images/tiff',480)


    # --- Configuration in order to resize rgb images ---
    #resize_resolution_maintaining_aspect_ratio(480,'/home/jacobo/Downloads/mines_dataset/images/train')

    # --- Configuration for 80/20 shuffling and splitting of a dataset ---
    #split_and_shuffle_dataset('/home/jacobo/Downloads/mines_multichannel_dataset/images/train','/home/jacobo/Downloads/mines_multichannel_dataset/labels/train','/home/jacobo/Downloads/mines_multichannel_dataset/images/val','/home/jacobo/Downloads/mines_multichannel_dataset/labels/val')



    # --- Configuration for visualizing a single tiff image ---
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



    # --- Configuration for Conversion from tiff to yolo11 png compatible dataset (this function expects a tiff dataset build by the fuse function)---
    #input_base_dir = '/home/jacobo/dataset/mines_multichannel_dataset/'
    #output_base_dir = '/home/jacobo/dataset/mines_multichannel_dataset_converted_png/'

    #input_image_subdir = 'images/train'
    #output_image_subdir = 'images/train'  # Keeping same subdirectory structure

    #input_val_subdir = 'images/val'
    #output_val_subdir = 'images/val'

    #convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset(input_base_dir, output_base_dir,input_image_subdir, output_image_subdir,input_val_subdir, output_val_subdir)


    #Straightforward conversion after having organized all the exr files and jpg files into two different directories; where each jpg is associated with it's corresponding exr file having the same names

    # 1. Calculate global min/max log-depth for the entire UNSPLIT dataset
    # This ensures consistent depth normalization across the final train/val/test splits.
    print(f"Starting global depth range calculation on all raw images...")

    all_rgb_files = sorted(
        [f for f in os.listdir('/home/jacobo/Downloads/mines_dataset_old/images/train') if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

    global_min_log_depth, global_max_log_depth = calculate_log_depth_global_min_max(
        rgb_src_files_list=all_rgb_files,
        depth_src_dir='/home/jacobo/Downloads/mines_dataset_old/images/fixed_exr_files'
    )



    # 2. Process and Fuse ALL raw RGB/EXR into 4-channel PNGs in a temporary single directory
    processed_count = process_and_fuse_all_to_png(
        rgb_src_dir='/home/jacobo/Downloads/mines_dataset_old/images/train',
        depth_src_dir='/home/jacobo/Downloads/mines_dataset_old/images/fixed_exr_files',
        labels_src_dir='/home/jacobo/Downloads/mines_dataset_old/labels/train',
        temp_output_base_dir='/home/jacobo/Downloads/test_new_fuse',
        global_min_log_depth=global_min_log_depth,
        global_max_log_depth=global_max_log_depth,
        TARGET_WIDTH=480
    )

    if processed_count == 0:
        print("No images were processed. Exiting without splitting.")
        exit()

    #3. shuffle and split 80/20
    split_and_shuffle_dataset('/home/jacobo/Downloads/test_new_fuse/images/train','/home/jacobo/Downloads/test_new_fuse/labels/train','/home/jacobo/Downloads/test_new_fuse/images/val','/home/jacobo/Downloads/test_new_fuse/labels/val')