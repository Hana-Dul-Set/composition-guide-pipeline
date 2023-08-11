from csnet import inference_from_dir
from file_utils import *

def blackbox(image_dir_path):

    assessed_image_list = inference_from_dir(image_dir_path)

    write_data(assessed_image_list)

    max_score_image = max(assessed_image_list, key=lambda x: x["score"])
    name_with_max_score = max_score_image["image_name"]
    return name_with_max_score

if __name__ == '__main__':
    blackbox('./cache')