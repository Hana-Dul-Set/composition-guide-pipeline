from image_edit_utils import *

OUTPUT_IMAGE_DIR = './cache/sliced_image/'

def get_input_image_dir_path(image_path):
    output_image_dir = save_all_sliced_image(input_image_path=image_path, sliced_image_dir=OUTPUT_IMAGE_DIR)

    return output_image_dir