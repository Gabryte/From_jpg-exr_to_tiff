from lib.main_functions import find_correct_exr_and_fix_it
from lib.utility import get_input_directories


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    array_of_exr_and_jpg_dirs = get_input_directories()

    find_correct_exr_and_fix_it("/home/jacobo/Downloads/mines_dataset/images/train",array_of_exr_and_jpg_dirs,'/home/jacobo/Downloads/mines_dataset/images/fixed_exr_files')


