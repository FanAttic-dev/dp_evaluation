from functools import cached_property
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import re

from tvcalib.inference import image_path2image_id
from utils.argsparse import EvalArgsNamespace, parse_args
from utils.config import Config
from utils.visualization import compare_view, show_overlap


class Evaluator:
    EPS = 1e-6
    WARPED_PATTERN = "*_warped.jpg"

    def __init__(self, args: EvalArgsNamespace):
        self.args = args
        self.var_path = Config.var_path
        self.main_path = Config.main_path

        assert self.var_path.exists() and self.main_path.exists()

        self.df = pd.read_csv(
            self.var_path / "losses.csv", index_col="image_id")
        self.n_var = len(self.df)
        self.th_low, self.th_high = self.thresholds
        self.main_paths = [""] * self.n_var

        self.ious = np.empty(self.n_var)
        self.ious.fill(np.nan)

        self.n_skipped = 0
        self.n_processed = 0
        self.iou_total = 0

    @cached_property
    def thresholds(self):
        q1, q3 = self.df["loss_ndc_total"].quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    @staticmethod
    def threshold(im) -> np.ndarray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)
        return im

    def evaluate(self):
        frame_folders = self.get_frame_folders()
        for main_frame_folder, var_frame_folder in frame_folders:
            self.process_frame_folders(main_frame_folder, var_frame_folder)

        self.save_csv()
        self.print_info()

    def get_frame_folders(self):
        var_frame_folders = sorted(
            [f for f in self.var_path.iterdir() if f.is_dir()]
        )

        frame_folders = []
        for var_folder in var_frame_folders:
            start, end = re.split("var_", var_folder.name)
            main_folders = [
                f for f in self.main_path.iterdir()
                if re.match(rf"{start}.*{end}", f.name)
            ]
            assert len(main_folders) == 1
            frame_folders.append((main_folders[0], var_folder))

        return frame_folders

    def process_frame_folders(self, main_frame_folder: Path, var_frame_folder: Path):
        main_imgs = sorted(main_frame_folder.glob(Evaluator.WARPED_PATTERN))
        var_imgs = sorted(var_frame_folder.glob(Evaluator.WARPED_PATTERN))

        for main_im_path, var_im_path in zip(main_imgs, var_imgs):
            var_id = image_path2image_id(
                var_im_path
            ).replace("_warped", "")

            df_idx = self.df.index == var_id
            df_idx_i = df_idx.nonzero()[0].item()
            self.main_paths[df_idx_i] = image_path2image_id(main_im_path)
            loss = self.df[df_idx]["loss_ndc_total"].item()
            if loss > self.th_high or loss < self.th_low:
                self.ious[df_idx] = np.nan
                self.n_skipped += 1
                print(f"[{var_id}]: Skipping with loss: {loss}")
                continue

            main_im = cv2.imread(str(main_im_path))
            var_im = cv2.imread(str(var_im_path))

            main_mask = self.threshold(main_im)
            var_mask = self.threshold(var_im)

            h, w = main_mask.shape
            intersection_mask = np.zeros((h, w, 1))
            intersection_mask[
                np.logical_and(main_mask > 0, var_mask > 0)
            ] = 1

            intersection = intersection_mask.sum()
            union = main_mask.sum() + var_mask.sum() - intersection
            iou = intersection / (union + Evaluator.EPS)
            self.ious[df_idx] = iou
            self.iou_total += iou

            print(f"[{var_id}]: IoU = {iou}")
            if args.show:
                compare_view(main_im, var_im)
                show_overlap(main_mask, var_mask, iou)
                plt.show()

            self.n_processed += 1

    def save_csv(self):
        self.df["image_main"] = self.main_paths
        self.df["iou"] = self.ious
        self.df.to_csv(self.main_path / "evaluation.csv")

    def print_info(self):
        print(
            f"Finished with {self.n_skipped} skipped and {self.n_processed} processed.\n \
                th_low: {self.th_low}, th_high: {self.th_high}\n \
                Avg IoU: {np.nanmean(self.ious)}, Median IoU: {np.nanmedian(self.ious)}"
        )


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(args)
    evaluator.evaluate()
