import os
import Imath
import OpenEXR
import numpy as np

from lib.utility import generate_file_hash, copy_and_rename_file


def load_single_channel_exr_map(exr_path):
    """
    Loads a single-channel floating-point map (e.g., depth, disparity) from an EXR file.

    This function is designed to robustly load depth or similar single-channel
    data from OpenEXR files. It prioritizes common depth channel names ('Z', 'depth')
    but can fall back to the 'R' (Red) channel if no explicit depth channel is found.
    It ensures the loaded NumPy array is writable and handles problematic
    floating-point values (NaNs and Infs) by converting them to zero.

    Args:
        exr_path (str): The full path to the EXR file to be loaded.

    Returns:
        numpy.ndarray or None: A 2D NumPy array of `float32` containing the loaded
                               channel data if successful. Returns `None` if the
                               file cannot be loaded, a suitable channel is not found,
                               or an error occurs during processing.
    """
    try:
        # 1. Open the EXR file
        exr_file = OpenEXR.InputFile(exr_path)

        # 2. Get data window and calculate image dimensions
        # 'dataWindow' defines the bounding box of the actual image data within the file.
        dw = exr_file.header()['dataWindow']
        # Calculate width and height from the data window.
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        # 3. Define channel name priorities for loading
        # These are common channel names for depth maps in EXR files.
        channel_names_priority = ['Z', 'depth', 'Depth', 'z']
        # Fallback channels, used if no priority channels are found. 'R' is often
        # used as the default channel for single-channel grayscale data.
        channel_names_fallback = ['R']

        selected_channel_name = None # Variable to store the name of the channel to be loaded
        # Get a list of all available channel names in the EXR file's header.
        available_channels = exr_file.header()['channels'].keys()

        # 4. Select the best available channel
        # First, try to find a channel from the high-priority list.
        for channel_name in channel_names_priority:
            if channel_name in available_channels:
                selected_channel_name = channel_name
                break # Found a priority channel, stop searching

        # If no priority channel was found, try the fallback channels.
        if selected_channel_name is None:
            for channel_name in channel_names_fallback:
                if channel_name in available_channels:
                    selected_channel_name = channel_name
                    break # Found a fallback channel, stop searching

        # If no suitable channel (priority or fallback) was found, print an error and return None.
        if selected_channel_name is None:
            print(f"Error: No recognized depth or fallback channel found in {exr_path}. "
                  f"Available channels: {available_channels}")
            return None

        # 5. Read the selected channel data
        # Define the pixel type for reading (float).
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        # Read the raw byte data for the selected channel.
        data = exr_file.channel(selected_channel_name, pixel_type)
        # Convert the raw bytes to a NumPy array of float32 and reshape it
        # to the correct image dimensions (height, width).
        img_map_raw = np.frombuffer(data, dtype=np.float32).reshape(size[1], size[0])

        # 6. Make a writable copy
        # The `np.frombuffer` operation might return a read-only array.
        # Creating a copy ensures that subsequent operations (like setting NaNs/Infs to 0)
        # can modify the array without errors.
        img_map = img_map_raw.copy()

        # 7. Handle potential infinite or NaN values
        # Infinite values (positive or negative) can occur in depth maps due to
        # sensor limitations or invalid measurements (e.g., very distant objects, no data).
        # We replace them with 0. You might choose a different value based on context
        # (e.g., a known maximum depth, or average depth).
        img_map[np.isinf(img_map)] = 0
        # NaN (Not-a-Number) values also represent invalid data.
        # We replace them with 0 to ensure numerical stability for further processing.
        img_map[np.isnan(img_map)] = 0

        # 8. Optional: Verify if NaNs/Infs were fully handled (for debugging)
        # This check is useful during development to ensure the cleanup steps are effective.
        # It prints a warning if any non-finite values unexpectedly remain after the cleanup.
        if np.any(np.isnan(img_map)) or np.any(np.isinf(img_map)):
            print(f"WARNING: NaNs/Infs still present after cleanup in {exr_path}! "
                  "This indicates an unexpected issue in data or cleanup logic.")

        return img_map

    except Exception as e:
        # Catch any exceptions that occur during file opening, reading, or processing.
        # Print an informative error message and return None to indicate failure.
        print(f"Failed to load EXR file '{exr_path}': {e}")
        return None


def find_missing_exr_for_jpg(jpg_directory, exr_directory):
    """
    Identifies JPG files in a directory that do not have a corresponding
    EXR file (with the same base name) in another directory.

    Args:
        jpg_directory (str): The path to the directory containing JPG files.
        exr_directory (str): The path to the directory containing EXR files.

    Returns:
        list: A list of full paths to JPG files that are missing their
              corresponding EXR file.
    """
    missing_jpg_files = []

    # Get all JPG filenames (without extension)
    jpg_base_names = set()
    for filename in os.listdir(jpg_directory):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            basename = os.path.splitext(filename)[0]
            jpg_base_names.add(basename)

    # Get all EXR filenames (without extension)
    exr_base_names = set()
    for filename in os.listdir(exr_directory):
        if filename.lower().endswith(".exr"):
            basename = os.path.splitext(filename)[0]
            exr_base_names.add(basename)

    # Find JPG basenames that are not in EXR basenames
    missing_basenames = jpg_base_names - exr_base_names

    # Reconstruct the full paths of the missing JPG files
    if missing_basenames:
        for filename in os.listdir(jpg_directory):
            current_basename = os.path.splitext(filename)[0]
            if current_basename in missing_basenames:
                missing_jpg_files.append(os.path.join(jpg_directory, filename))

    return missing_jpg_files


def check_exr_content(exr_file_path):
    """
    Inspects an OpenEXR file to display its header information and attempts
    to read and analyze a specific channel ('R') if it exists.

    This function serves as a diagnostic tool to verify the contents and
    structure of an EXR file, including its available channels, data window,
    compression, and other metadata. It also demonstrates how to extract pixel
    data from a single channel and convert it into a NumPy array for further

    Args:
        exr_file_path (str): The file path to the .exr file to be inspected.
    """
    # Use a try-except block to gracefully handle potential I/O errors,
    # such as the file not being found or being improperly formatted.
    try:
        # Open the specified EXR file in read-only mode.
        file = OpenEXR.InputFile(exr_file_path)

        # Retrieve the file's header, which is a dictionary containing all metadata.
        header = file.header()

        print(f"Header for {exr_file_path}:")
        # Iterate through all key-value pairs in the header and print them for review.
        for key, value in header.items():
            # The 'channels' value is a dictionary of channel objects.
            # We print just the channel names (e.g., 'R', 'G', 'B', 'A') for concise output.
            if key == 'channels':
                print(f"  Channels: {list(value.keys())}")
            else:
                print(f"  {key}: {value}")

        # --- Example: Reading and Analyzing a Specific Channel ---
        # Check if an 'R' (red) channel exists in the file's channel list.
        if 'R' in header['channels']:
            # Specify the data type for reading the pixel data (e.g., 32-bit float).
            data_type = Imath.PixelType(Imath.PixelType.FLOAT)

            # Read the raw byte data for the 'R' channel. The result is a byte string.
            r_slice = file.channels('R', data_type)[0]

            # Get the data window from the header, which defines the bounding box of
            # the pixel data within the image plane.
            dw = header['dataWindow']

            # Calculate the width and height of the image from the data window coordinates.
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

            # Convert the raw byte buffer into a 1D NumPy array of 32-bit floats,
            # then reshape it into a 2D array (height, width) matching the image dimensions.
            r_data = np.frombuffer(r_slice, dtype=np.float32).reshape(size[1], size[0])

            # Print basic statistics (min/max values) of the channel data to confirm
            # it was read correctly and to understand its value range.
            print(f"\nSuccessfully read 'R' channel. Min value: {r_data.min()}, Max value: {r_data.max()}")

            # Optional: The following commented-out code can be enabled to visualize
            # the channel data as a grayscale image using matplotlib. This is very
            # useful for visually inspecting depth maps or other single-channel data.
            # import matplotlib.pyplot as plt
            # plt.imshow(r_data, cmap='gray'); plt.colorbar(); plt.title("R Channel Data"); plt.show()
        else:
            print("\nNo 'R' channel found in the file.")

    except Exception as e:
        # If any error occurs during the file opening or reading process,
        # print a descriptive error message to the console.
        print(f"Error inspecting EXR file {exr_file_path}: {e}")


def find_correct_exr_and_fix_it(dataset_jpg_dir, array_of_jpg_and_exr_dirs, exr_output_dir):
    """
    Identifies and copies the correct EXR depth maps based on matching JPG images' content.

    This function compares JPG images from a primary 'dataset' directory with JPG images
    found within various 'no_dataset' directories. When a JPG image in the 'dataset'
    directory has an identical content hash to a JPG image in a 'no_dataset' directory,
    its corresponding EXR depth map (found in the 'depth' subdirectory of the
    'no_dataset' structure) is copied to a specified output directory. The copied
    EXR file is renamed to match the base name of the 'dataset' JPG image.

    Args:
        dataset_jpg_dir (str): The path to the directory containing the "ground truth"
                               or reference JPG images. These images are hashed to find
                               matching counterparts.
        array_of_jpg_and_exr_dirs (list): A list of directory paths. Each directory in
                                         this list is expected to contain two specific
                                         subdirectories:
                                         - 'rgb/': For JPG images that potentially match
                                                   the 'dataset_jpg_dir' images.
                                         - 'depth/': For EXR depth map images that
                                                     correspond to the JPGs in 'rgb/'.
        exr_output_dir (str): The path to the directory where the correctly identified
                              and renamed EXR files will be copied. This directory will
                              be created if it doesn't exist.
    """

    # Dictionary to store hashes of JPG images from the dataset directory.
    # Key: File hash (str), Value: Full path to the JPG file (str).
    jpg_dataset_hashes = {}
    # Dictionary to store hashes of JPG images from the 'no_dataset' directories
    # and their corresponding EXR file paths.
    # Key: JPG file hash (str), Value: Full path to the associated EXR file (str).
    jpg_no_dataset_hashes = {}

    print(f"Scanning dataset directory: {dataset_jpg_dir} for JPG images and their hashes...")
    # Iterate through all files in the 'dataset_jpg_dir'.
    for filename in os.listdir(dataset_jpg_dir):
        # Check if the file is a JPG/JPEG and not a hidden file.
        if filename.lower().endswith((".jpg", ".jpeg")) and not filename.startswith("."):
            file_path = os.path.join(dataset_jpg_dir, filename)

            # Ensure the path points to an actual file before processing.
            if os.path.isfile(file_path):
                # Generate a hash for the JPG file's content.
                file_hash = generate_file_hash(file_path)
                # Store the hash and the file path in the dataset hashes dictionary.
                jpg_dataset_hashes[file_hash] = file_path
            else:
                # This case might occur if os.listdir returns directories or broken symlinks.
                print(f"Skipping '{file_path}' as it is not a valid JPG file or does not exist.")


    print(f"Scanning source directories for JPG-EXR pairs: {array_of_jpg_and_exr_dirs}...")
    # Iterate through each parent directory provided in `array_of_jpg_and_exr_dirs`.
    # Each 'father_dir' is expected to contain 'rgb' and 'depth' subdirectories.
    for father_dir in array_of_jpg_and_exr_dirs:
        print(f"Processing source directory: {father_dir}...")
        # Construct the path to the 'rgb' subdirectory within the current father_dir.
        single_no_dataset_jpg_dir = os.path.join(father_dir, "rgb")

        # Iterate through the JPG files within the 'rgb' subdirectory.
        for filename in os.listdir(single_no_dataset_jpg_dir):
            # Check if the file is a JPG/JPEG and not a hidden file.
            if filename.lower().endswith((".jpg", ".jpeg")) and not filename.startswith("."):
                # Construct the path to the 'depth' subdirectory within the current father_dir.
                single_exr_dir = os.path.join(father_dir, "depth")
                # Extract the base name of the JPG file (without extension).
                file_name_without_ext = os.path.splitext(filename)[0]
                # Construct the full path to the corresponding EXR file.
                exr_file_path = os.path.join(single_exr_dir, file_name_without_ext + ".exr")
                # Construct the full path to the current JPG file from the 'no_dataset' source.
                no_dataset_jpg_file_path = os.path.join(single_no_dataset_jpg_dir, filename)

                # Verify that both the JPG and its corresponding EXR file exist before processing.
                if os.path.isfile(no_dataset_jpg_file_path) and os.path.isfile(exr_file_path):
                    # Generate a hash for the content of the 'no_dataset' JPG file.
                    file_hash = generate_file_hash(no_dataset_jpg_file_path)
                    # Store the JPG hash and the path to its associated EXR file.
                    jpg_no_dataset_hashes[file_hash] = exr_file_path
                else:
                    print(f"Skipping processing for JPG '{no_dataset_jpg_file_path}' and EXR '{exr_file_path}' "
                          f"as one or both files do not exist.")

    print('Starting comparison of JPG files to find matching EXR depth maps...')
    # Now, compare the hashes from the dataset JPGs with the hashes from the 'no_dataset' JPGs.
    # This loop iterates through each JPG hash found in the primary 'dataset_jpg_dir'.
    for hash_val, dataset_jpg_path in jpg_dataset_hashes.items():
        # If the hash of a dataset JPG is found in the 'no_dataset' JPG hashes,
        # it means we've identified a content-identical JPG.
        if hash_val in jpg_no_dataset_hashes:
            print(f"Match found! Dataset JPG: '{dataset_jpg_path}' (Hash: {hash_val[:8]}...) "
                  f"matches a source JPG with corresponding EXR: '{jpg_no_dataset_hashes[hash_val]}'")

            # Ensure the output directory for EXR files exists. If not, create it.
            os.makedirs(exr_output_dir, exist_ok=True)

            # Get the base filename of the matched JPG from the dataset.
            # This name will be used to rename the copied EXR file.
            new_name = os.path.basename(dataset_jpg_path)

            # Extract the filename without its extension (e.g., "image123" from "image123.jpg").
            new_true_base_name = os.path.splitext(new_name)[0]
            # Construct the new filename for the EXR file (e.g., "image123.exr").
            new_true_name = new_true_base_name + ".exr"

            # Retrieve the full path to the EXR file corresponding to the matched JPG hash.
            exr_source_filepath = jpg_no_dataset_hashes[hash_val]
            # Get the original filename of the EXR file.
            old_name = os.path.basename(exr_source_filepath)
            # Get the directory of the original EXR file.
            exr_source_path = os.path.dirname(exr_source_filepath)

            # Perform the copy operation only if the source EXR file actually exists.
            if os.path.isfile(exr_source_filepath):
                print(f"Attempting to copy EXR file from '{exr_source_filepath}' to '{exr_output_dir}' as '{new_true_name}'...")
                try:
                    # Call the helper function to copy and rename the EXR file.
                    copy_and_rename_file(exr_source_path, exr_output_dir, old_name, new_true_name)
                except FileNotFoundError:
                    # Handle cases where the source file might unexpectedly not be found during the copy attempt.
                    print(f"Error: Source EXR file not found at '{exr_source_filepath}' during copy operation.")
                except Exception as e:
                    # Catch any other potential errors during the file copying process.
                    print(f"An unexpected error occurred during EXR file copying: {e}")
            else:
                # This should ideally not happen if the previous os.path.isfile check passed,
                # but it serves as an additional safeguard.
                print(f"Copy Failed: Source EXR file '{exr_source_filepath}' unexpectedly not found.")
        else:
            print(f"No matching EXR found for dataset JPG: '{dataset_jpg_path}' (Hash: {hash_val[:8]}...)")

    print("EXR finding and fixing process completed.")
