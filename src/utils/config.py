from pathlib import Path

import numpy as np
import utils.utils as utils


class Config:
    @staticmethod
    def load_pitch_corners(corners_path):
        corners_dict = utils.load_yaml(corners_path)
        pts = np.array(
            [[v["x"], v["y"]] for v in corners_dict.values()],
            dtype=np.int32
        )
        return pts.reshape((-1, 1, 2))

    config = utils.load_yaml("./configs/config_eval.yaml")
    var_path = Path(config["dataset"]["var_path"])
    main_path = Path(config["dataset"]["main_path"])
    masks_output_dir = Path(config["masks_output_dir"])
    image_size = config["image_size"]
    export_int_sec = config["export_every_x_seconds"]

    assets_path = Path('assets')
    pitch_model_path = assets_path / 'pitch_model.png'
    pitch_model_corners_path = assets_path / 'pitch_model_corners.yaml'
    pitch_corners = load_pitch_corners(pitch_model_corners_path)
