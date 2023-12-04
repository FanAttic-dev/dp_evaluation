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
from utils.utils import save_to_file
from utils.visualization import compare_view, show_overlap


class Evaluator:
    EPS = 1e-6
    WARPED_PATTERN = "*_warped.jpg"
    LOSS_COL = "loss_ndc_total"

    def __init__(self, args: EvalArgsNamespace):
        self.var_path = Config.var_path
        self.main_path = Config.main_path
        assert self.var_path.exists() and self.main_path.exists()

        self.args = args
        self.csv_path_var = self.var_path / "losses.csv"
        self.csv_path_main = self.main_path / "evaluation.csv"
        self.txt_result_path = self.main_path / "result.txt"
        self.figures_path = self.main_path / "iou_figures"
        self.is_evaluated = self.load_csv()
        self.figure = plt.figure(figsize=(6, 8))

    def load_csv(self):
        if self.csv_path_main.exists():
            print("evaluation.csv already present")
            self.df = pd.read_csv(
                self.csv_path_main,
                index_col="image_id"
            )

            self.ious = self.df["iou"]
            self.main_paths = self.df["image_main"]

            self.th_low, self.th_high = self.thresholds
            self.n_var = len(self.df)

            self.n_processed = len(self.df[
                (~self.df["iou"].isnull())
                # (self.df[Evaluator.LOSS_COL] > self.th_low) &
                # (self.df[Evaluator.LOSS_COL] < self.th_high)
            ])
            self.n_skipped = self.n_var - self.n_processed
            self.iou_total = self.df["iou"].sum()
            return True

        self.df = pd.read_csv(
            self.csv_path_var,
            index_col="image_id"
        )

        self.n_var = len(self.df)
        self.th_low, self.th_high = self.thresholds

        self.ious = np.empty(self.n_var)
        self.ious.fill(np.nan)
        self.main_paths = [""] * self.n_var

        self.n_processed = 0
        self.n_skipped = 0
        self.iou_total = 0
        return False

    @cached_property
    def thresholds(self):
        q1, q3 = self.df[Evaluator.LOSS_COL].quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    @staticmethod
    def threshold(im) -> np.ndarray:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, im = cv2.threshold(im, 1, 1, cv2.THRESH_BINARY)
        return im

    def evaluate(self):
        frame_folders = self.get_frame_folders()
        for main_frame_folder, var_frame_folder in frame_folders:
            self.process_frame_folders(main_frame_folder, var_frame_folder)

        self.save_csv()

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
            loss = self.df[df_idx][Evaluator.LOSS_COL].item()
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
            intersection_mask = np.zeros((h, w))
            intersection_mask[
                np.logical_and(main_mask > 0, var_mask > 0)
            ] = 1

            union_mask = main_mask + var_mask - intersection_mask
            iou = intersection_mask.sum() / (union_mask.sum() + Evaluator.EPS)
            self.ious[df_idx] = iou
            self.iou_total += iou

            print(f"[{var_id}]: IoU = {iou}")

            if args.show or args.fig_save:
                fig = self.make_figure(
                    main_im, var_im, main_mask, var_mask, iou)

            if args.show:
                plt.show(block=False)
                key = input("Press Enter to continue or Ctrl+C+Enter to exit.")

            if args.fig_save:
                self.save_figure(
                    fig, clip_name=main_frame_folder.stem, i=df_idx_i)

            self.n_processed += 1

    def make_figure(self, main_im, var_im, main_mask, var_mask, iou):
        fig = self.figure
        plt.cla()
        plt.clf()
        grid = plt.GridSpec(2, 2)
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1])
        ax3 = fig.add_subplot(grid[1, :])

        compare_view(main_im, var_im, [ax1, ax2])
        show_overlap(main_mask, var_mask, iou, ax3)

        plt.tight_layout(pad=1)
        return fig

    def save_figure(self, fig, clip_name, i):
        Path.mkdir(self.figures_path, exist_ok=True)
        path = self.figures_path / f"{clip_name}_{i:04d}"
        fig.savefig(path)

    def save_csv(self):
        self.df["image_main"] = self.main_paths
        self.df["iou"] = self.ious
        self.df.to_csv(self.csv_path_main)

    def print_info(self):
        self.df["directory"] = self.df.image_main.apply(
            lambda path: Path(
                path).parts[-2] if not pd.isnull(path) and len(path) > 0 else None
        )
        df_grouped = self.df.groupby("directory")
        df_info = pd.DataFrame({
            "iou_mean": df_grouped["iou"].mean(),
            "iou_median": df_grouped["iou"].median(),
        })

        info = f"""
Finished with {self.n_skipped} skipped and {self.n_processed} processed.
    th_low: {self.th_low:.9f}
    th_high: {self.th_high:.9f}
    
"""

        info += str(df_info)
        info += f"""
        
Overall iou_mean: {np.nanmean(self.ious):.9f}
Overall iou_median: {np.nanmedian(self.ious):.9f}
"""
        save_to_file(self.txt_result_path, info)
        print(info)


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(args)
    if not evaluator.is_evaluated or args.show:
        evaluator.evaluate()
    evaluator.print_info()
