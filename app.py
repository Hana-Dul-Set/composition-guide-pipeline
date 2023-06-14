from flask import Flask, request, jsonify
from flask_cors import CORS

from input_utils import *
from blackbox import *
from output_utils import *

import os

app = Flask(__name__)


@app.route('/guide', methods=['POST'])
def composition_guide():
    image_file = request.files['image']
    image_name = image_file.filename

    if not os.path.exists('cache'):
        os.mkdir('cache')
    image_path = './cache/' + image_name
    image_file.save(image_path)
    print(image_path)

    ## 슬라이싱하여 저장
    image_path = get_input_image_dir_path(image_path)

    ## 추론 후 최대가 되는 이미지 식별
    blackbox_output = blackbox(image_path)

    ## 이동방향
    camera_forwward_direction = get_camera_forward_direction(blackbox_output)

    response = {
        'direction': camera_forwward_direction
    }

    return jsonify(response)

@app.route('/test')
def hello():
    return 'python'

if __name__ == '__main__':
    app.run(port=8000)
