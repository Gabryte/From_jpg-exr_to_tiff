from lib.global_dataset_utility import calculate_global_min_max, split_and_shuffle_dataset
from lib.main_functions import find_correct_exr_and_fix_it
from lib.utility import get_input_directories
from lib.utility import find_missing_exr_for_jpg
from lib.main_functions import fuse
from lib.global_dataset_utility import resize_resolution_maintaining_aspect_ratio
# Press the green button in the gutter to run the script.
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

    split_and_shuffle_dataset('/home/jacobo/Downloads/mines_dataset/images/train','/home/jacobo/Downloads/mines_dataset/labels/train','/home/jacobo/Downloads/mines_dataset/images/val','/home/jacobo/Downloads/mines_dataset/labels/val')



