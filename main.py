import numpy as np

from tvcalib_wrapper import TVCalibWrapper

tvcalib_wrapper = TVCalibWrapper()
image_ids, keypoints_raw = tvcalib_wrapper.segment()
df = tvcalib_wrapper.calibrate(image_ids, keypoints_raw)
print(df.iloc[0].homography)
