#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rasterio

from transforms import calculate_fdi, calculate_ndvi
from skimage.exposure import equalize_hist
from data import l2abands, l1cbands
import matplotlib
import matplotlib.cm
import subprocess
import os
import pandas as pd
import glob

def write(arr, filename, profile):
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write((arr * 255).astype(rasterio.uint8))

def process_s2tiff(tiff, target_folder="/tmp/viz", upload=True, convert=True, thumbnail=True):
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


    rgb, ndvi, fdi, meta = load_convert_tiff(tiff)

    name, ext = os.path.splitext(os.path.basename(tiff))
    profile = meta
    profile.update(dict(
        count=3,
        dtype="uint8"
    ))

    def convert_upload(arr, type):

        fname = name + "_" + type + ext
        filename = os.path.join(target_folder, "tif", fname)
        cogfilename = os.path.join(target_folder, "cog", fname)
        thumbname = os.path.join(target_folder, "thumb", name + "_" + type + ".jpg")

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs(os.path.dirname(cogfilename), exist_ok=True)
        os.makedirs(os.path.dirname(thumbname), exist_ok=True)

        write(arr, filename, profile)

        print("convert to COG")
        print(f"gdalwarp {filename} {cogfilename} -of COG -overwrite")
        if convert:
            subprocess.call(["gdalwarp", filename, cogfilename, "-of", "COG", "-overwrite"])
        print("upload to bucket:")
        print(f"aws s3 cp {cogfilename} s3://floatingobjects/data/ --acl public-read")
        url = f"https://floatingobjects.s3.eu-central-1.amazonaws.com/data/{fname}"
        expl_url = f"https://geotiffjs.github.io/cog-explorer/#scene={url}&bands=&pipeline="
        if upload:
            subprocess.call(["aws", "s3", "cp", cogfilename, "s3://floatingobjects/data/", "--acl", "public-read"])
            print("tif should now be available at")
            print(expl_url)
        cmd = ["gdal_translate", "-of", "JPEG", "-b", "1", "-b", "2", "-b", "3", filename, thumbname, "-outsize", "100", "100"]
        print(" ".join(cmd))
        if thumbnail:
            subprocess.call(cmd)

        return f"[![{name}](doc/thumb/{os.path.basename(thumbname)})]({expl_url})"

    name_cell = "`" + name + "`"
    ndvi_cell = convert_upload(ndvi[:3], "ndvi")
    fdi_cell = convert_upload(fdi[:3], "fdi")
    rgb_cell = convert_upload(rgb, "rgb")

    return [name_cell, rgb_cell, ndvi_cell, fdi_cell]

def process_prediction(tiff, target_folder = "/tmp/pred", convert=True, upload=True, thumbnail=True):
    with rasterio.open(tiff) as src:
        arr = src.read()
        if src.meta["dtype"] == "uint8":
            arr = arr * 2**-8
        elif src.meta["dtype"] == "uint16":
            arr = arr * 2 ** -16
        meta = src.meta

    cmap = matplotlib.cm.get_cmap('magma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    pred = cmap(norm(arr[0]))
    pred = np.rollaxis(pred, axis=2)[:3]

    os.makedirs(target_folder, exist_ok=True)
    profile = meta
    profile.update(
        dict(
            count=3,
            dtype="uint8"
        )
    )

    name, ext = os.path.splitext(os.path.basename(tiff))
    fname = name + "_pred" + ext
    filename = os.path.join(target_folder, "tif", fname)
    cogfilename = os.path.join(target_folder, "cog", fname)
    thumbname = os.path.join(target_folder, "thumb", name + "_pred" + ".jpg")

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    os.makedirs(os.path.dirname(cogfilename), exist_ok=True)
    os.makedirs(os.path.dirname(thumbname), exist_ok=True)

    write(pred, filename, profile)

    print("convert to COG")
    print(f"gdalwarp {filename} {cogfilename} -of COG -overwrite")
    if convert:
        subprocess.call(["gdalwarp", filename, cogfilename, "-of", "COG", "-overwrite"])
    print("upload to bucket:")
    print(f"aws s3 cp {cogfilename} s3://floatingobjects/predictions/ --acl public-read")
    url = f"https://floatingobjects.s3.eu-central-1.amazonaws.com/predictions/{fname}"
    expl_url = f"https://geotiffjs.github.io/cog-explorer/#scene={url}&bands=&pipeline="
    if upload:
        subprocess.call(["aws", "s3", "cp", cogfilename, "s3://floatingobjects/predictions/", "--acl", "public-read"])
        print("tif should now be available at")
        print(expl_url)
    cmd = ["gdal_translate", "-of", "JPEG", "-b", "1", "-b", "2", "-b", "3", filename, thumbname, "-outsize", "100", "100"]
    print(" ".join(cmd))
    if thumbnail:
        subprocess.call(cmd)

    return [f"[![{name}](doc/thumb/{os.path.basename(thumbname)})]({expl_url})"]

def create_dummy_image(meta):

    count = meta["count"]
    width = meta["width"]
    height = meta["height"]

    imagepath = "/tmp/dummyimage.tif"
    with rasterio.open(imagepath, "w", **meta) as dst:
        dummydata = (np.random.rand(count, height, width) * (2 ** 16)).astype("uint16")
        dst.write(dummydata)

    return imagepath

def get_test_images(pred_path):
    test_images = glob.glob(os.path.join(pred_path, "*", "test", "*.tif"))
    test_names = [os.path.basename(f).replace(".tif", "") for f in test_images]

    df = pd.DataFrame(dict(path=test_images, name=test_names)).set_index("name")
    # keep first in duplicates
    df = df[~df.index.duplicated(keep='first')]
    return df

if __name__ == '__main__':

    """
    count = 1
    incrs = rasterio.crs.CRS.from_epsg(32636)
    intransform = rasterio.transform.Affine(10.0, 0.0, 502590.0,
                                            0.0, -10.0, 3837650.0)

    imgmeta = {'driver': 'GTiff',
               'dtype': 'uint16',
               'nodata': None,
               'width': 200,
               'height': 200,
               'count': count,
               'crs': incrs,
               'transform': intransform}

    tiff = create_dummy_image(imgmeta)
    tiff = "/ssd2/floatingObjects/predictions/0/test/danang_20181005.tif"
    process_prediction(tiff, upload=True, convert=True)
    """
    print()
    pred_path = "/ssd2/floatingObjects/predictions"
    data_path = "/ssd2/floatingObjects/data"
    target_folder = "/tmp/viz"
    convert = True
    upload = True

    df = get_test_images(pred_path)

    df = df.sort_index()

    rows = list()
    for name, row in df.iterrows():
        path = row.path

        tiff = os.path.join(data_path, name + ".tif")

        cells = process_s2tiff(tiff, target_folder=target_folder, upload=upload, convert=convert, thumbnail=True)
        cells += process_prediction(path, target_folder=target_folder, convert=convert, upload=upload, thumbnail=True)

        row = " | " + " | ".join(cells) + " | "
        rows.append(row)

    print("\n".join(rows))



