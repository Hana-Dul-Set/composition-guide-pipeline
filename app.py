from flask import Flask, request, jsonify
from flask_cors import CORS

from input_utils import *
from blackbox import *
from output_utils import *

import os, shutil, random
from time import time, strftime, localtime

app = Flask(__name__)


@app.route('/guide', methods=['POST'])
def composition_guide():
    time_list = []
    print('1. ', time())
    time_list.append(time())
    image_file = request.files['image']
    ext = image_file.filename.split('.')[-1]
    image_name = strftime('%Y%m%d_%H:%M:%S_', localtime(time())) + str(random.randrange(1, 100)) + '.' + ext
    print('2. ', time())
    time_list.append(time())
    if not os.path.exists('cache'):
        os.mkdir('cache')
    image_dir_path = './cache'
    image_path = os.path.join(image_dir_path, image_name)
    image_file.save(image_path)
    print(image_path, '!!!!')
    print('3. ', time())
    time_list.append(time())
    ## 슬라이싱하여 저장
    input_image_dir_path = get_input_image_dir_path(image_path)
    print('4. ', time())
    time_list.append(time())
    ## 추론 후 최대가 되는 이미지 식별
    blackbox_output = blackbox(input_image_dir_path)
    print('5. ', time())
    time_list.append(time())
    ## 이동방향
    camera_forwward_direction = get_camera_forward_direction(blackbox_output)
    print(camera_forwward_direction)
    print('6. ', time())
    time_list.append(time())
    ## 받은 이미지와 슬라이싱한 이미지 삭제
    """
    os.remove(image_path)
    sliced_image_list = os.listdir(input_image_dir_path)
    for image in sliced_image_list:
        os.remove(input_image_dir_path + image)
    """

    ## 처리된 이미지들 data 폴더에 옮기기
    os.remove(image_path)
    sliced_image_list = os.listdir(input_image_dir_path)
    for image_name in sliced_image_list:
        image_path = os.path.join(input_image_dir_path, image_name)
        shutil.move(image_path, "./data/images")
    print('7. ', time())
    time_list.append(time())
    response = {
        'direction': camera_forwward_direction
    }
    print('8. ', time())
    time_list.append(time())
    for i in range(0, len(time_list)-1):
        print(i+1, time_list[i+1]-time_list[i], '\n')
    return jsonify(response)

@app.route('/test')
def hello():
    return 'python'

if __name__ == '__main__':
    app.run(port=8000)
