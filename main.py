import matplotlib.pyplot as plt
from tvcalib_wrapper import TVCalibWrapper
from pathlib import Path
import numpy as np
import os
import sys
sys.path.append('./tvcalib') if os.path.isdir('./tvcalib') else ''

images_path = Path("data/datasets/test-autocam")
output_dir = Path("tmp")

tvcalib_wrapper = TVCalibWrapper(images_path, output_dir)
image_ids, keypoints_raw = tvcalib_wrapper.segment()
df = tvcalib_wrapper.calibrate(image_ids, keypoints_raw)

img_warped = tvcalib_wrapper.warp_frame(df.iloc[0], overlay=True)
