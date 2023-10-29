from pathlib import Path
import utils.utils as utils


class Config:
    config = utils.load_yaml("./configs/config_eval.yaml")
    var_path = Path(config["dataset"]["var_path"])
    main_path = Path(config["dataset"]["main_path"])
    image_size = config["image_size"]
    export_int_sec = config["export_every_x_seconds"]
