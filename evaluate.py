from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tvcalib.inference import image_path2image_id


def compare_view(main_im, var_im):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))
    imgs = [("Main", main_im), ("VAR", var_im)]

    for i, (title, im) in enumerate(imgs):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax[i].imshow(im)
        ax[i].set_title(title)
        ax[i].axis('off')

    plt.subplots_adjust(hspace=0.01, wspace=0.01,
                        left=0.01, bottom=0.01, right=0.99, top=0.99)
    plt.show()


def threshold(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, im = cv2.threshold(im, 1, 255, cv2.THRESH_BINARY)
    return im


def show_overlap(main_im, var_im):
    assert main_im.shape == var_im.shape
    h, w = main_im.shape

    colors = {
        "Main": (1, 0, 0),  # yellow
        "VAR": (1, 1, 0),  # turquoise
        "Overlap": (0, 1, 0)  # green
    }
    im_cameras = np.zeros((h, w, 3))
    im_cameras[main_im > 0] = colors["Main"]
    im_cameras[var_im > 0] = colors["VAR"]

    im_overlap = np.zeros((h, w, 3))
    im_overlap[np.logical_and(main_im > 0, var_im > 0)] = colors["Overlap"]

    im = cv2.addWeighted(im_cameras, 0.7, im_overlap, 1, 0)

    figure, ax = plt.subplots()
    ax.set_title("Overlapped")
    ax.imshow(im)

    patches = [mpatches.Patch(color=np.array(v)*0.9, label=k)
               for k, v in colors.items()]
    ax.legend(handles=patches, loc="lower right")

    plt.show()


var_path = Path("../../datasets/TrnavaZilina/VAR")
main_path = Path("../dp_autocam/recordings/MATCH/2023-09-26")

assert var_path.exists() and main_path.exists()

LOSS_THRESHOLD = 0.02
EPS = 1e-6

df = pd.read_csv(var_path / "losses.csv", index_col="image_id")

warped_pattern = "*_warped.jpg"
ious = np.zeros(len(df))
for period in ["p0", "p1"]:
    main_folder = main_path / f"main_{period}_frames"
    var_folder = var_path / f"var_{period}_frames"

    main_imgs = sorted(main_folder.glob(warped_pattern))
    var_imgs = sorted(var_folder.glob(warped_pattern))

    n = min(len(var_imgs), len(main_imgs))
    n_skipped = 0
    n_processed = 0
    iou_total = 0
    for i in range(n):
        main_im_path = main_imgs[i]
        var_im_path = var_imgs[i]

        var_id = image_path2image_id(var_im_path).replace("_warped", "")
        df_idx = df.index == var_id
        loss = df[df_idx]["loss_ndc_total"].item()
        if loss > LOSS_THRESHOLD:
            ious[df_idx] = np.nan
            n_skipped += 1
            print(f"[{var_id}]: Skipping with loss: {loss}")
            continue

        main_im = cv2.imread(str(main_im_path))
        var_im = cv2.imread(str(var_im_path))
        # compare_view(main_im, var_im)

        main_mask = threshold(main_im)
        var_mask = threshold(var_im)
        # show_overlap(main_im_th, var_im_th)

        h, w = main_mask.shape
        intersection_mask = np.zeros((h, w, 1))
        intersection_mask[np.logical_and(main_mask > 0, var_mask > 0)] = 1

        intersection = intersection_mask.sum()
        union = main_mask.sum() + var_mask.sum() - intersection
        iou = intersection / (union + EPS)
        ious[df_idx] = iou
        iou_total += iou

        print(f"[{var_id}]: IoU = {iou}")

        n_processed += 1

iou_avg = iou_total / n_processed
df["iou"] = ious
df.to_csv(var_path / "losses_iou.csv")
print(
    f"Finished with {n_skipped} skipped and {n_processed} processed. Avg IoU: {iou_avg}")
