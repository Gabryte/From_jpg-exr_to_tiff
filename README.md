To use this repository:
 
 0. You will need the exported frames (Jpg + Exr) according to the default structure to which Record3D exports and the CVAT exported (after you have annotated it) dataset
 
 1. Clone it
 
 2. You can install the requirements.txt located under the lib folder ( pip install -r requirements.txt )
 
 3. You will need to modify, with your IDE, the file paths (in main.py) needed to merge exr and jpg frames etc...
 
 4. To run main.py you can either use your IDE or the command line

Procedure to obtain the Yolo 4 channels tiff dataset (the following code is already present in main.py):

 0. Prepare the main.py functions
 
 1. ```
    array_of_exr_and_jpg_dirs = get_input_directories('/home/jacobo/Desktop/Video RGB+D Florence/') #stop the path when you reach EXR_RGBD subdirectory
    ```
 
 2. ```
    find_correct_exr_and_fix_it("/home/jacobo/Downloads/CVAT_EXPORTED_DATASET/images/train",array_of_exr_and_jpg_dirs,'/home/jacobo/Downloads/CVAT_EXPORTED_DATASET/images/fixed_exr_files')
    ``` 
  
  2.1. The first path points to the jpg images obtained via CVAT exportation.
  
  2.2. The second parameter is a mandatory array of paths used to track those exr files linked to their corresponding jpg files.  
  
  2.3. The third parameter is the desired location in which you can store the linked (by the same name) exr files to their corresponeding jpg files.
 
 3. 
    ```
      #1. Calculate global min/max log-depth for the entire UNSPLIT dataset
      This ensures consistent depth normalization across the final train/val/test splits.
      print(f"Starting global depth range calculation on all raw images...")
  
      all_rgb_files = sorted(
          [f for f in os.listdir('/home/jacobo/Downloads/CVAT_EXPORTED_DATASET/images/train') if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
  
      global_min_log_depth, global_max_log_depth = calculate_log_depth_global_min_max(
          rgb_src_files_list=all_rgb_files,
          depth_src_dir='/home/jacobo/Downloads/CVAT_EXPORTED_DATASET/images/fixed_exr_files'
      )
  
      #2.1 Process and Fuse ALL raw RGB/EXR into 4-channel TIFFs <- it works either for the 4 channel training and for the saving of the best.pt in 4 channel format
      processed_count = process_and_fuse_all_to_tiff(
          rgb_src_dir='/home/jacobo/Downloads/CVAT_EXPORTED_DATASET/images/train',
          depth_src_dir='/home/jacobo/Downloads/CVAT_EXPORTED_DATASET/images/fixed_exr_files',
          labels_src_dir='/home/jacobo/Downloads/CVAT_EXPORTED_DATASET/labels/train',
          temp_output_base_dir='/home/jacobo/Downloads/new_train_tiff',
          global_min_log_depth=global_min_log_depth,
          global_max_log_depth=global_max_log_depth,
          TARGET_WIDTH=480
      )
  
      if processed_count == 0:
          print("No images were processed. Exiting without splitting.")
          exit()
  
      #3. shuffle and split 80/20
      split_and_shuffle_dataset('/home/jacobo/Downloads/new_train_tiff/images/train','/home/jacobo/Downloads/new_train_tiff/labels/train','/home/jacobo/Downloads/new_train_tiff/images/val','/home/jacobo/Downloads/new_train_tiff/labels/val')
    ```

