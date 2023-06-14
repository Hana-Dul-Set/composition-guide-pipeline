import csv

def write_data(assessed_image_list):
    with open('./data/assessment_result.csv', 'a') as csv_file:
        fieldnames = ['image_name', 'pred_score', 'pattern_weight_list', 'attribute_weight_list']
        csv_writer = csv.DictWriter(csv_file, fieldnames)

        for image_data in assessed_image_list:
            csv_writer.writerow(image_data)

    return