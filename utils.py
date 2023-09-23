import numpy as np
import json


def coords_to_pts(coords):
    pts = np.array([[v["x"], v["y"]] for v in coords.values()], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)
