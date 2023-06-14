import cv2
import matplotlib.pyplot as plt
import os

SLCIE_RATIO = 0.167

def get_image_height(image):
    return len(image)

def get_image_width(image):
    return len(image[0])

def get_top_image(image):
    image_height = get_image_height(image)
    image_width = get_image_width(image)
    image_finish_height = int(image_height - image_height * SLCIE_RATIO)
    sliced_image = image[0:image_finish_height, 0:image_width].copy()
    return sliced_image

def get_bottom_image(image):
    image_height = get_image_height(image)
    image_width = get_image_width(image)
    image_start_height = int(image_height * SLCIE_RATIO)
    sliced_image = image[image_start_height:image_height, 0:image_width].copy()
    return sliced_image

def get_left_image(image):
    image_height = get_image_height(image)
    image_width = get_image_width(image)
    image_finish_width = int(image_width - image_width * SLCIE_RATIO)
    sliced_image = image[0:image_height, 0:image_finish_width].copy()
    return sliced_image

def get_right_image(image):
    image_height = get_image_height(image)
    image_width = get_image_width(image)
    image_start_width = int(image_width * SLCIE_RATIO)
    sliced_image = image[0:image_height, image_start_width:image_width].copy()
    return sliced_image

def save_all_sliced_image(input_image_path, sliced_image_dir):
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image_name = input_image_path.split('/')[-1].split('.')[0]
    image_ext = '.' + input_image_path.split('/')[-1].split('.')[-1]

    source_image = image
    top_image = get_top_image(image)
    bottom_image = get_bottom_image(image)
    left_image = get_left_image(image)
    right_image = get_right_image(image)
    if not os.path.exists(sliced_image_dir):
        os.mkdir(sliced_image_dir)
    print(sliced_image_dir + image_name + '_s_' + image_ext, "$$$$$")
    cv2.imwrite(sliced_image_dir + image_name + '_s_' + image_ext, source_image)
    cv2.imwrite(sliced_image_dir + image_name + '_t_' + image_ext, top_image)
    cv2.imwrite(sliced_image_dir + image_name + '_b_' + image_ext, bottom_image)
    cv2.imwrite(sliced_image_dir + image_name + '_l_' + image_ext, left_image)
    cv2.imwrite(sliced_image_dir + image_name + '_r_' + image_ext, right_image)


    return sliced_image_dir

if __name__ == '__main__':
    save_all_sliced_image('./cache/i_eiffel_1.jpg', './cache/')
