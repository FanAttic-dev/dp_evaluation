import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def compare_view(main_im, var_im, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4.5))

    imgs = [("Main", main_im), ("VAR", var_im)]

    for i, (title, im) in enumerate(imgs):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax[i].imshow(im)
        ax[i].set_title(title)
        ax[i].set_axis_off()

    plt.subplots_adjust(hspace=0.01, wspace=0.01,
                        left=0.01, bottom=0.01, right=0.99, top=0.99)


def show_overlap(main_im, var_im, iou, ax=None):
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

    if ax is None:
        figure, ax = plt.subplots()
    ax.set_title(f"IoU: {iou:.6f}")
    ax.set_axis_off()
    ax.imshow(im)

    patches = [mpatches.Patch(color=np.array(v)*0.9, label=k)
               for k, v in colors.items()]
    ax.legend(handles=patches, loc="center",
              bbox_to_anchor=(0.5, -0.08), ncol=3)
