from collections import defaultdict
import random
from argparse import Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import numpy as np
import kornia
import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import SoccerPitch
import time


from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_tribune import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate, split_circle_central
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output
from tvcalib.utils import visualization_mpl_min as viz

args = Namespace(
    images_path=Path("data/datasets/test-autocam"),
    output_dir=Path("tmp"),
    checkpoint="data/segment_localization/train_59.pt",
    gpu=True,
    nworkers=1,
    batch_size_seg=32,
    batch_size_calib=32,
    image_width=1280,
    image_height=720,
    optim_steps=1000,
    lens_dist=True,
    write_masks=True
)
device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

object3d = SoccerPitchLineCircleSegments(
    device=device, base_field=SoccerPitchSNCircleCentralSplit()
)

dataset_seg = InferenceDatasetSegmentation(
    args.images_path, args.image_width, args.image_height
)
print("number of images:", len(dataset_seg))
dataloader_seg = torch.utils.data.DataLoader(
    dataset_seg,
    batch_size=args.batch_size_seg,
    num_workers=args.nworkers,
    shuffle=False,
    collate_fn=custom_list_collate,
)
