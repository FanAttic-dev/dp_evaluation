import matplotlib.pyplot as plt
from tvcalib_wrapper import TVCalibWrapper
from pathlib import Path
import numpy as np
import os
import sys
sys.path.append('./tvcalib') if os.path.isdir('./tvcalib') else ''

# images_path = Path("data/datasets/test-autocam")
images_path = Path("../../datasets/TrnavaZilina/VAR")
output_dir = Path("tmp")

tvcalib_wrapper = TVCalibWrapper(images_path, output_dir)

# Segment
image_ids, keypoints_raw = tvcalib_wrapper.segment()

# Calibrate
df = tvcalib_wrapper.calibrate(image_ids, keypoints_raw)

# Warp & save
for idx, sample in df.iterrows():
    img_warped = tvcalib_wrapper.warp_frame(sample, overlay=True)
    tvcalib_wrapper.save_frame(sample, img_warped)
