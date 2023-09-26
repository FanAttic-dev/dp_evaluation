from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

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


var_path = Path("../../datasets/TrnavaZilina/VAR")
main_path = Path("../dp_autocam/recordings")

LOSS_THRESHOLD = 0.02

df = pd.read_csv(var_path / "losses.csv")

warped_pattern = "*_warped.jpg"
for period in ["p0", "p1"]:
    var_folder = var_path / f"var_{period}_frames"
    main_folder = main_path / f"main_{period}_frames"

    var_imgs = sorted(var_folder.glob(warped_pattern))
    main_imgs = sorted(main_folder.glob(warped_pattern))

    n = min(len(var_imgs), len(main_imgs))
    n_skipped = 0
    n_processed = 0
    for i in range(n):
        var_im_path = var_imgs[i]
        main_im_path = main_imgs[i]

        var_id = image_path2image_id(var_im_path).replace("_warped", "")
        loss = df[df["image_id"] == var_id]["loss_ndc_total"].item()
        if loss > LOSS_THRESHOLD:
            n_skipped += 1
            # print(f"Skipping {var_id}, loss: {loss}")
            continue

        var_im = cv2.imread(str(var_im_path))
        main_im = cv2.imread(str(main_im_path))

        compare_view(main_im, var_im)

        n_processed += 1


print(f"Finished with {n_skipped} skipped and {n_processed} processed")
