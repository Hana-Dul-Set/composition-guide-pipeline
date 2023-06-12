from input_utils import *
from blackbox import *
from output_utils import *

if __name__ == '__main__':
    
    image = get_image()

    blackbox_output = blackbox(image)

    camera_forward_direction = get_camera_forward_direction(blackbox_output)
