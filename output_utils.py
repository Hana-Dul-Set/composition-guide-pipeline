DIRECTION_MAP = {
    'source': 'S',
    'Top': 'U',
    'Bottom': 'D',
    'Left': 'L',
    'Right': 'R'
}

def get_camera_forward_direction(blackbox_output):
    return DIRECTION_MAP[blackbox_output]
