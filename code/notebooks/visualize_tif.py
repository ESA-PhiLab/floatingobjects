import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib
from data import l2abands, l1cbands
from skimage.exposure import equalize_hist
from transforms import calculate_fdi, calculate_ndvi


def acquire_data(file_name):
    """Read an L1C Sentinel-2 image from a cropped TIF. The image is represented as TOA reflectance.
    Args:
        file_name (str): event ID.
    Raises:
        ValueError: impossible to find information on the database.
    Returns:
        np.array: array containing B8A, B11, B12 of a Seintel-2 L1C cropped tif.
        dictionary: dictionary containing lat and lon for every image point.
    """

    with rasterio.open(file_name) as raster:
        img_np = raster.read()
        sentinel_img = img_np.astype(np.float32)
        height = sentinel_img.shape[1]
        width = sentinel_img.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
        lons = np.array(ys)
        lats = np.array(xs)
        coords_dict = {"lat": lats, "lon": lons}

    sentinel_img = (
        sentinel_img.transpose(1, 2, 0) / 10000 + 1e-13
    )  # Diving for the default quantification value

    return sentinel_img, coords_dict


def load_convert_tiff(tiff):

    with rasterio.open(tiff) as src:
        arr = src.read()
        meta = src.meta

    if arr.shape[0] == 12:
        bands = l2abands
    elif arr.shape[0] == 13:
        bands = l1cbands
    else:
        raise ValueError("expected tiff to have either 12 (L2A) or 13 (L1C) bands")

    rgb = equalize_hist(arr[[bands.index("B4"), bands.index("B3"), bands.index("B2")]])

    cmap_magma = matplotlib.cm.get_cmap('magma')
    cmap_viridis = matplotlib.cm.get_cmap('viridis')

    norm_fdi = matplotlib.colors.Normalize(vmin=0, vmax=0.1)
    norm_ndvi = matplotlib.colors.Normalize(vmin=-.4, vmax=0.4)

    ndvi = cmap_viridis(norm_ndvi(calculate_ndvi(arr)))
    ndvi = np.rollaxis(ndvi, axis=2)
    fdi = cmap_magma(norm_fdi(calculate_fdi(arr)))
    fdi = np.rollaxis(fdi, axis=2)
    return rgb, ndvi, fdi, meta


if __name__ == "__main__":
    file_name = "../../data2/data/london_20180611.tif"
    sentinel_img, coords_dict = acquire_data(file_name)
    for i in range(0, 11):
        print(i)
    #    plt.imshow(sentinel_img[:, :, i:i+3])
    #    plt.show()
    print(coords_dict['lat'].shape)

    rgb, ndvi, fdi, meta = load_convert_tiff(file_name)
    plt.imshow(np.moveaxis(rgb, 0, -1))
    plt.show()

