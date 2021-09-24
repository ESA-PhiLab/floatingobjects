import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_hist
from itertools import cycle
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


def plot_curves(nets, fpr, tpr, roc_auc, recall, prec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    lw = 2
    colors = cycle(['aqua', 'red']) if len(nets) == 2 else cycle(['aqua', 'red', 'cornflowerblue', 'moccasin', 'mediumpurple'])

    # ROC Curves
    if "svm" in fpr:
        ax1.plot(fpr["svm"], tpr["svm"], label='Support-Vector Machine (AUC = {0:0.2f})'.format(roc_auc["svm"]), color='deeppink', linewidth=lw)
        ax1.plot(fpr["rf"], tpr["rf"], label='Random Forest (AUC = {0:0.2f})'.format(roc_auc["rf"]), color='navy', linewidth=lw)
        ax1.plot(fpr["nb"], tpr["nb"], label='Naïve Bayes (AUC = {0:0.2f})'.format(roc_auc["nb"]), color='forestgreen', linewidth=lw)
        ax1.plot(fpr["hgb"], tpr["hgb"], label='Hist. Gradient Boosting (AUC = {0:0.2f})'.format(roc_auc["hgb"]), color='darkorange', linewidth=lw)

    for i, color in zip(nets, colors):
        j = 'U-Net' if i=='unet-cross-val-2fold' else 'MA-Net'
        ax1.plot(fpr[j], tpr[j], color=color, lw=lw, label='{0} (AUC = {1:0.2f})'.format(j, roc_auc[j]))
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01], 
            xlabel='False Positive Rate', ylabel='True Positive Rate', 
            title='Receiver Operating Characteristic')

    # Recall/Precision Curves
    if "svm" in fpr:
        ax2.plot(recall["svm"], prec["svm"], label='Support-Vector Machine', color='deeppink', linewidth=lw)
        ax2.plot(recall["rf"], prec["rf"], label='Random Forest', color='navy', linewidth=lw)
        ax2.plot(recall["nb"], prec["nb"], label='Naïve Bayes', color='forestgreen', linewidth=lw)
        ax2.plot(recall["hgb"], prec["hgb"], label='Hist. Gradient Boosting', color='darkorange', linewidth=lw)

    for i, color in zip(nets, colors):
        j = 'U-Net' if i=='unet-cross-val-2fold' else 'MA-Net'
        ax2.plot(recall[j], prec[j], color=color, lw=lw, label=f'{j}')
    ax2.set(xlim=[-0.01, 1.01], ylim=[-0.01, 1.01], 
            xlabel='Recall', ylabel='Precision', 
            title='Precision/Recall curve')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)
    return fig