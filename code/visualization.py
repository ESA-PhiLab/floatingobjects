
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_hist

from data import l2abands as bands

def calculate_fdi(scene):
    # tbd

    NIR = scene[bands.index("B8")] * 1e-4
    RED2 = scene[bands.index("B6")] * 1e-4
#    RED2 = cv2.resize(RED2, NIR.shape)

    SWIR1 = scene[bands.index("B11")] * 1e-4
    #SWIR1 = cv2.resize(SWIR1, NIR.shape)

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    return NIR - NIR_prime

def s2_to_RGB(scene):
    tensor = np.stack([scene[bands.index('B4')],scene[bands.index('B3')],scene[bands.index('B2')]])
    return equalize_hist(tensor.swapaxes(0,1).swapaxes(1,2))

def ndvi_transform(scene):
    NIR = scene[bands.index("B8")]
    RED = scene[bands.index("B4")]
    return (NIR - RED) / (NIR + RED + 1e-12)

def plot_batch(images, masks, y_preds):
    N = images.shape[0]

    height = 3
    width = 3
    fig, axs = plt.subplots(N, 5, figsize=(5 * width, N * height))
    for axs_row, img, mask, y_pred in zip(axs, images, masks, y_preds):
        axs_row[0].imshow(s2_to_RGB(img), cmap="magma")
        axs_row[0].set_title("RGB")
        axs_row[1].imshow(ndvi_transform(img), cmap="viridis")
        axs_row[1].set_title("NDVI")
        axs_row[2].imshow(calculate_fdi(img), cmap="magma")
        axs_row[2].set_title("FDI")
        axs_row[3].imshow(mask)
        axs_row[3].set_title("Mask")
        axs_row[4].imshow(y_pred)
        axs_row[4].set_title("Prediction")
        [ax.axis("off") for ax in axs_row]
    return fig