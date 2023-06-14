SOURCE_IMAGE_NAME = ''

DIRECTION_MAP = {
    's': 'S',
    't': 'U',
    'b': 'D',
    'l': 'L',
    'r': 'R'
}

def get_camera_forward_direction(blackbox_output):
    tag = blackbox_output.split('_')[-2]
    return DIRECTION_MAP[tag]
