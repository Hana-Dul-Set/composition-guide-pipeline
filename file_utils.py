import csv, os, random, shutil
from time import time, strftime, localtime

def move_used_images(request_image_path, input_image_dir_path):
    os.remove(request_image_path)
    shutil.move(request_image_path, input_image_dir_path)
    
    input_image_list = os.listdir(input_image_dir_path)
    for image_name in input_image_list:
        image_path = os.path.join(input_image_dir_path, image_name)
        shutil.move(image_path, "./data/images")

def save_request_image(request):
    image_file = request.files['image']
    ext = image_file.filename.split('.')[-1]
    image_name = strftime('%Y%m%d_%H:%M:%S_', localtime(time())) + str(random.randrange(1, 100)) + '.' + ext

    if not os.path.exists('cache'):
        os.mkdir('cache')
    image_dir_path = './cache'
    image_path = os.path.join(image_dir_path, image_name)
    image_file.save(image_path)

    return image_path

def write_data(assessed_image_list):
    with open('./data/assessment_result.csv', 'a') as csv_file:
        fieldnames = ['image_name', 'score', 'pattern_weight_list', 'attribute_weight_list']
        csv_writer = csv.DictWriter(csv_file, fieldnames)

        for image_data in assessed_image_list:
            csv_writer.writerow(image_data)

    return