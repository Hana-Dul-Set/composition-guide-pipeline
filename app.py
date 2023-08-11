import os, shutil, random
from time import time, strftime, localtime

from flask import Flask, request, jsonify
from flask_cors import CORS

from input_utils import *
from blackbox import *
from output_utils import *
from file_utils import *

app = Flask(__name__)

@app.route('/guide', methods=['POST'])
def composition_guide():

    request_image_path = save_request_image(request)

    ## 블랙박스에 넣을 수 있도록 input 처리
    input_image_dir_path = get_input_image_dir_path(request_image_path)

    ## 블랙박스에 넣고 ouput 받기
    blackbox_output = blackbox(input_image_dir_path)

    ## output을 바탕으로 방향 반환
    camera_forwward_direction = get_camera_forward_direction(blackbox_output)
    print(camera_forwward_direction)

    remove_used_images(request_image_path, input_image_dir_path)

    response = {
        'direction': camera_forwward_direction
    }

    return jsonify(response)

@app.route('/test')
def hello():
    return 'python'

if __name__ == '__main__':
    app.run(port=8000)
