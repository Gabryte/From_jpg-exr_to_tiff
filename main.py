from lib.evaluation import analyze_model_errors_rgbd
from lib.global_dataset_utility import calculate_global_min_max, split_and_shuffle_dataset, process_and_convert_images, \
    calculate_log_depth_global_min_max, shuffle_frames_randomly, split_dataset_arbitrary_percentages
from lib.fusion_related_functions import find_correct_exr_and_fix_it, \
    convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset, process_and_fuse_all_to_png, \
    process_and_fuse_all_to_tiff
from lib.utility import get_input_directories, visualize_multichannel_tiff, check_exr_content
from lib.utility import find_missing_exr_for_jpg
from lib.fusion_related_functions import fuse
from lib.global_dataset_utility import resize_resolution_maintaining_aspect_ratio
import OpenEXR
import Imath
import numpy as np
import os
if __name__ == '__main__':


    # --- Configuration needed in order to create a single training rgb dataset that have all of it's corresponding exr files associated via the same name. It's essential for the next step of fusion between rgb and exr frames ---

    #Getting all the different directories of the exported images in jpg and exr files. This process is mandatory in order to get an array of paths, where each of them points to the directory that contains two subfolders: depth and rgb (automatically produced by the Record3D app when exporting in jpg and exr)

    #array_of_exr_and_jpg_dirs = get_input_directories('/home/jacobo/Desktop/Video RGB+D Florence/Annotated')

    #find_correct_exr_and_fix_it("/home/jacobo/Downloads/train_dataset_cvat_export/images/train",array_of_exr_and_jpg_dirs,'/home/jacobo/Downloads/train_dataset_cvat_export/images/fixed_exr_files')

    # --- Configuration in order to find eventually missing exr files for rgb images due to iphone 16 export errors ---
    #missing = find_missing_exr_for_jpg("/home/jacobo/Downloads/mines_dataset/images/train",'/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files')

    # --- Configuration of the old fuse function (it produces tiff images in a format that yolo don't understand) ---
    #fuse('/home/jacobo/Downloads/mines_dataset/images/train','/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files','/home/jacobo/Downloads/mines_dataset/images/tiff',480)


    # --- Configuration in order to resize images ---
    #resize_resolution_maintaining_aspect_ratio(480,'/home/jacobo/Downloads/mines_dataset/images/train')


    # --- Configuration for 80/20 shuffling and splitting of a dataset ---
    #split_and_shuffle_dataset('/home/jacobo/Downloads/mines_multichannel_dataset/images/train','/home/jacobo/Downloads/mines_multichannel_dataset/labels/train','/home/jacobo/Downloads/mines_multichannel_dataset/images/val','/home/jacobo/Downloads/mines_multichannel_dataset/labels/val')



    # --- Configuration for visualizing a single tiff image with depths in float32 (this function expects tiff images produced by the old fuse() function) ---
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



    # --- Configuration for Conversion from tiff to yolo11 png incompatible dataset (this function expects a tiff dataset build by the fuse function) ---
    #input_base_dir = '/home/jacobo/dataset/mines_multichannel_dataset/'
    #output_base_dir = '/home/jacobo/dataset/mines_multichannel_dataset_converted_png/'

    #input_image_subdir = 'images/train'
    #output_image_subdir = 'images/train'  # Keeping same subdirectory structure

    #input_val_subdir = 'images/val'
    #output_val_subdir = 'images/val'

    #convert_from_tiff_multichannel_dataset_to_yolo11_multichannel_dataset(input_base_dir, output_base_dir,input_image_subdir, output_image_subdir,input_val_subdir, output_val_subdir,'labels/Test')


    #  --- Configuration for straightforward conversion after having organized all the exr files and jpg files into two different directories; where each jpg is associated with it's corresponding exr file having the same names ---

    # 1. Calculate global min/max log-depth for the entire UNSPLIT dataset
    # This ensures consistent depth normalization across the final train/val/test splits.
    #print(f"Starting global depth range calculation on all raw images...")

    #all_rgb_files = sorted(
    #    [f for f in os.listdir('/home/jacobo/Downloads/test_set_exported_cvat/images/Test') if f.lower().endswith(('.jpeg', '.jpg', '.png'))])

    #global_min_log_depth, global_max_log_depth = calculate_log_depth_global_min_max(
    #    rgb_src_files_list=all_rgb_files,
    #    depth_src_dir='/home/jacobo/Downloads/test_set_exported_cvat/images/fixed_exr_files'
    #)

    #print(f"min: {global_min_log_depth}, max: {global_max_log_depth}")
    #On train Global Min Log Depth: 0.0, Global Max Log Depth: 2.505526065826416
    #On test Global Min Log Depth: -0.01998910680413246, Global Max Log Depth: 2.2355434894561768


    # 2.1 Process and Fuse ALL raw RGB/EXR into 4-channel TIFFs <- it works either for the 4 channel training and for the saving of the best.pt in 4 channel format
   # processed_count = process_and_fuse_all_to_tiff(
   #     rgb_src_dir='/home/jacobo/Downloads/test_set_exported_cvat/images/train',
   #     depth_src_dir='/home/jacobo/Downloads/test_set_exported_cvat/images/fixed_exr_files',
   #     labels_src_dir='/home/jacobo/Downloads/test_set_exported_cvat/labels/train',
   #     temp_output_base_dir='/home/jacobo/Downloads/fused_rgbd_test_images',
   #     global_min_log_depth=-0.01998910680413246,
   #     global_max_log_depth=2.505526065826416,
   #     TARGET_WIDTH=600
   # )

    #if processed_count == 0:
        #print("No images were processed. Exiting without splitting.")
        #exit()

    #3. shuffle and split 80/20 only train and val
    #split_and_shuffle_dataset('/home/jacobo/Downloads/new_train_tiff/images/train','/home/jacobo/Downloads/new_train_tiff/labels/train','/home/jacobo/Downloads/new_train_tiff/images/val','/home/jacobo/Downloads/new_train_tiff/labels/val')
    #3.1 IMPORTANT Alternatively you can use this function in order to split with custom percentages train val and test
    # Example Usage:

    # Define your paths (assuming they exist or the script will create them for non-zero percentages)
    #train_images_dir = "/home/jacobo/Downloads/advanced_rgbd_dataset/images/train"
    #train_labels_dir = "/home/jacobo/Downloads/advanced_rgbd_dataset/labels/train"
    #val_images_dir = "/home/jacobo/Downloads/advanced_rgbd_dataset/images/val"
    #val_labels_dir = "/home/jacobo/Downloads/advanced_rgbd_dataset/labels/val"
    #test_images_dir = "/home/jacobo/Downloads/advanced_rgbd_dataset/images/test"
    #test_labels_dir = "/home/jacobo/Downloads/advanced_rgbd_dataset/labels/test"

    # Example: 80-10-10 split
    #print("\n--- Running 80-10-10 split ---")
    #split_dataset_arbitrary_percentages(
    #    train_images_dir, train_labels_dir,
    #    val_images_dir, val_labels_dir,
    #    test_images_dir, test_labels_dir,
    #    train_percentage=80, val_percentage=20, test_percentage=0
    #)

    # --- Configuration for the shuffling of the test set (use it only if you have prepared the dataset with train val and not test)---
    #shuffle_frames_randomly('/home/jacobo/Downloads/test_rgbd/images/Test','/home/jacobo/Downloads/test_rgbd/labels/Test')

    # --- Configuration for checking the globals max and mins using the non logP function ---
    # List all RGB frames (assuming they are named sequentially, e.g., 00000.jpg)
    #rgb_files_old = sorted([f for f in os.listdir('/home/jacobo/Downloads/mines_dataset_old/images/train') if f.endswith(('.jpeg', '.jpg'))])
    #rgb_files_for_rgbd = sorted([f for f in os.listdir('/home/jacobo/Downloads/test_dataset_rgbd/images/Test') if f.endswith(('.jpeg', '.jpg'))])

    #min_train,max_train = calculate_global_min_max(rgb_files_old,'/home/jacobo/Downloads/mines_dataset_old/images/fixed_exr_files/')
    #min_test,max_test = calculate_global_min_max(rgb_files_for_rgbd, '/home/jacobo/Downloads/test_dataset_rgbd/images/fixed_exr_files/')



    #print(f"min_train: {min_train}, max_train: {max_train}")
    #print(f"min_test: {min_test}, max_test: {max_test}")

    # --- Configuration for checking the content of an exr file
    #check_exr_content('/home/jacobo/Downloads/test_dataset_rgbd/images/fixed_exr_files/0_8.exr')

    # --- Configuration for an in-depth model error analysis
    # Define your paths
    model_path = '/home/jacobo/Downloads/my_yolo_train/107_epochs_classic_dataset_rgbd_train/weights/best.pt'  # e.g., 'runs/detect/train/weights/best.pt'
    test_images_dir = '/home/jacobo/Downloads/fused_rgbd_test_images/images/Test'
    test_labels_dir = '/home/jacobo/Downloads/fused_rgbd_test_images/labels/Test'
    test_depth_dir = '/home/jacobo/Downloads/test_set_exported_cvat/images/fixed_exr_files'  # Directory containing original EXR depth files
    output_analysis_dir = '/home/jacobo/Downloads/analysis_results'

    # You MUST provide these from your dataset preprocessing
    # These are the min/max log depths *calculated across your entire training dataset*.
    # If you don't have these, you'll need to compute them first.
    # For example, by iterating through your depth maps during dataset creation.
    # If your depth normalization is different, adjust `normalize_array_to_range` and depth estimation.
    global_min_log_depth = -0.01998910680413246  # Placeholder, replace with actual value
    global_max_log_depth = 2.505526065826416  # Placeholder, replace with actual value


    TARGET_WIDTH = 600  # Ensure this matches the image size your model was trained with

    analyze_model_errors_rgbd(
        model_path=model_path,
        test_images_dir=test_images_dir,
        test_labels_dir=test_labels_dir,
        test_depth_dir=test_depth_dir,
        output_dir=output_analysis_dir,
        iou_threshold=None,  # Standard IoU threshold for evaluation
        conf_threshold=None,  # Confidence threshold for displaying detections
        class_conf_thresholds={
            0:0.0525,
            1:0.04,
            2:0.05,
            3:0.05,
        },
        class_iou_thresholds={
            0:0.03,
            1:0.18,
            2:0.01,
            3:0.11
        },
        global_min_log_depth=global_min_log_depth,
        global_max_log_depth=global_max_log_depth,
        TARGET_WIDTH=TARGET_WIDTH
    )

