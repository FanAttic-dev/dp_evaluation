import numpy as np
import yaml


def coords_to_pts(coords):
    pts = np.array([[v["x"], v["y"]] for v in coords.values()], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        return yaml.safe_load(f)
