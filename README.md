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
      #3.1 IMPORTANT Alternatively you can use this function in order to split with custom percentages train val and test
      # Example Usage:
        
      # Define your paths (assuming they exist or the script will create them for non-zero percentages)
      #train_images_dir = "datasets/my_yolo_dataset/images/train"
      #train_labels_dir = "datasets/my_yolo_dataset/labels/train"
      #val_images_dir = "datasets/my_yolo_dataset/images/val"
      #val_labels_dir = "datasets/my_yolo_dataset/labels/val"
      #test_images_dir = "datasets/my_yolo_dataset/images/test"
      #test_labels_dir = "datasets/my_yolo_dataset/labels/test"
    
      # Example: 80-10-10 split
      #print("\n--- Running 80-10-10 split ---")
      #split_dataset_arbitrary_percentages(
      #    train_images_dir, train_labels_dir,
      #    val_images_dir, val_labels_dir,
      #    test_images_dir, test_labels_dir,
      #    train_percentage=80, val_percentage=10, test_percentage=10
      #)
    ```


If you would like to train a yolo model, you need a virtual enviroment installed on the training machine:

1. Update the system: Before installing, it's a good practice to update your system's package list and upgrade installed packages, *only if you have root privileges*.
```
   sudo apt update
   sudo apt upgrade
```
2. Download the Miniconda installer (Navigate to your user's home directory): 
```
   cd ~
```
3. Download the Miniconda installer for Linux from the official Miniconda releases page. You can use wget to download the installer: 
```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```
4. make the script executable. 
```
   chmod +x miniconda.sh
```
5. Run the installer (Execute the script using bash):
```
   bash miniconda.sh
```
*Follow the prompts during installation. You'll be asked to review the license agreement and choose an installation directory. Important --> Accept all of the defaults, expecially when conda asks for the automatic startup initialization.* 

6. Initialize Conda (After installation, you need to initialize Conda for it to be available in your terminal sessions): 
```
   conda init
```
*You might need to close and reopen your terminal or source the ~/.bashrc file for the changes to take effect.* 
```
   source ~/.bashrc
```
7. Verify the installation:
```
   conda --version
```
*You should see the Conda version number printed in the output. Then you need to create your virtual enviroment as it's a good practice to create a dedicated environment for your projects to avoid dependency conflicts, don't use the (base) conda enviroment because it's a system level enviroment.*
```
conda create -n yolov11 python=3.12
```
*-n yolov11: Specifies the name of your new environment (you can choose any name you like, e.g., yolov11_ultralytics).*

*python=3.12: Sets the Python version for this environment. You can adjust this if a different version is required, don't use python = 3.13 or superiors versions, because ultralytics doesn't support them.*

8. Before installing anything, you need to activate the environment you just created, in fact you will install all the python libraries on the miniconda environment that you created:
```
conda activate yolov11
```
*You should see the environment name in your terminal prompt (e.g., (yolov11)).*

9. Install PyTorch (with CUDA for an nvidia GPU)
*YOLOv11 relies on PyTorch. If you have an NVIDIA GPU, it's highly recommended to install the CUDA-enabled version of PyTorch for faster inference and training.*

Visit the official PyTorch website (https://pytorch.org/get-started/locally/) and select your preferences (OS, Package, Language, CUDA version). In our case Linux Ubuntu

A common command for CUDA 11.8 might look like this (verify the exact command on the PyTorch website as it changes):
*IMPORTANT to check the machine CUDA version run nvidia-smi*
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
10. Install Ultralytics
Once PyTorch is installed, you can install the Ultralytics package, which includes YOLOv11.
```
pip install ultralytics
```
11. Verify the Installation
You can quickly verify that YOLOv8 is installed and working by running a simple command:
```
yolo help
```
This should display the YOLO CLI help message, indicating that the installation was successful.


