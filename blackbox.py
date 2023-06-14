import sys, csv
sys.path.append('./SAMPNet')

from inference import *
from file_utils import *


def blackbox(image_dir_path):
    print(image_dir_path, "2222")
    assessed_image_list = inference_from_dir(image_dir_path)
    print(assessed_image_list, "3333")

    write_data(assessed_image_list)

    max_score_image = max(assessed_image_list, key=lambda x: x["score"])
    name_with_max_score = max_score_image["image_name"]
    return name_with_max_score

if __name__ == '__main__':
    blackbox('./cache')