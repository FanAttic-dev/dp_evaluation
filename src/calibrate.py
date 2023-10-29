import pandas as pd

from tvcalib_wrapper import TVCalibWrapper
from utils.config import Config

images_path = Config.var_path
output_dir = Config.masks_output_dir


def main():
    tvcalib_wrapper = TVCalibWrapper(images_path, output_dir)

    # Segment
    image_ids, keypoints_raw = tvcalib_wrapper.segment()

    # Calibrate
    df = tvcalib_wrapper.calibrate(image_ids, keypoints_raw)

    # Warp & save
    for idx, sample in df.iterrows():
        img_warped = tvcalib_wrapper.warp_frame(sample, Config.pitch_overlay)
        tvcalib_wrapper.save_frame(sample, img_warped)

    df_losses = pd.DataFrame(df, columns=[
        "homography", "loss_ndc_lines", "loss_ndc_circles", "loss_ndc_total"
    ])
    df_losses.to_csv(images_path / "losses.csv")


if __name__ == "__main__":
    main()
