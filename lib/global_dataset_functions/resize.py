import os

import cv2
from PIL import Image


def resize_resolution_maintaining_aspect_ratio(TARGET_WIDTH, IMAGES_PATH):
    """
    Resizes all images in a specified directory to a target width while maintaining
    their aspect ratio. Images are overwritten with their resized versions.
    It handles both downgrading and upgrading resolution. Hidden files (starting with '.')
    are skipped.

    Args:
        TARGET_WIDTH (int): The desired width for all images.
        IMAGES_PATH (str): The path to the directory containing image files.
    """
    # Validate IMAGES_PATH
    if not os.path.isdir(IMAGES_PATH):
        print(f"Error: IMAGES_PATH '{IMAGES_PATH}' does not exist or is not a directory.")
        return

    # Validate TARGET_WIDTH
    if not isinstance(TARGET_WIDTH, int) or TARGET_WIDTH <= 0:
        print(f"Error: TARGET_WIDTH must be a positive integer. Got '{TARGET_WIDTH}'.")
        return

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
    resized_count = 0

    print(f"\n--- Resizing images in '{IMAGES_PATH}' to a target width of {TARGET_WIDTH}px ---")

    for filename in os.listdir(IMAGES_PATH):
        full_path = os.path.join(IMAGES_PATH, filename)

        # Skip directories and hidden files
        if os.path.isdir(full_path) or filename.startswith('.'):
            # print(f"Skipping directory or hidden file: '{filename}'")
            continue

        # Check if it's a supported image file
        if not filename.lower().endswith(image_extensions):
            print(f"Skipping non-image file: '{filename}'")
            continue

        try:
            with Image.open(full_path) as img:
                original_width, original_height = img.size

                # Calculate new height maintaining aspect ratio
                aspect_ratio = original_height / original_width
                new_height = int(TARGET_WIDTH * aspect_ratio)

                # Resize the image using Image.Resampling.LANCZOS
                resized_img = img.resize((TARGET_WIDTH, new_height), Image.Resampling.LANCZOS)

                # Save the resized image, overwriting the original
                resized_img.save(full_path)
                resized_count += 1
                # print(f"Resized '{filename}' from {original_width}x{original_height} to {TARGET_WIDTH}x{new_height}.")

        except FileNotFoundError:
            print(f"Error: Image file not found during resize operation for '{filename}'. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{filename}': {e}. Skipping.")

    print(f"\n--- Image resizing complete! ---")
    print(f"Successfully resized {resized_count} images in '{IMAGES_PATH}'.")


def resize_rgb_and_depth_maintain_aspect_ratio(TARGET_WIDTH, rgb_frame, depth_map):
    """
    Resizes RGB and depth frames to a target width while maintaining their original aspect ratio.
    """
    original_height, original_width = rgb_frame.shape[0], rgb_frame.shape[1]

    # Calculate target height to maintain aspect ratio
    target_height = int(original_height * (TARGET_WIDTH / original_width))

    #print(f"Resizing to: {TARGET_WIDTH}x{target_height} (maintaining aspect ratio)")

    # Resize RGB (uint8)
    rgb_frame_resized = cv2.resize(rgb_frame, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA)

    # Resize Depth map (float32)
    depth_map_resized = cv2.resize(depth_map, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_LINEAR)

    return rgb_frame_resized, depth_map_resized
