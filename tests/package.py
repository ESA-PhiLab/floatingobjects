import os
import sys

this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder, "..", "code"))
import numpy as np

import unittest

from model import get_model
from predictor import PythonPredictor
import torch
from itertools import product
from train import snapshot, resume

import rasterio



def create_dummy_image(meta):
    count = meta["count"]
    width = meta["width"]
    height = meta["height"]

    imagepath = "/tmp/dummyimage.tif"
    with rasterio.open(imagepath, "w", **meta) as dst:
        dummydata = (np.random.rand(count, height, width) * (2 ** 16)).astype("uint16")
        dst.write(dummydata)

    return imagepath

class ModelTestCases(unittest.TestCase):

    def test_model(self):
        model = get_model("unet", inchannels=12)
        X = torch.ones(3, 12, 256, 256)
        y_pred = model(X)
        self.assertTrue(y_pred.shape == torch.Size([3, 1, 256, 256]))
        y_pred.mean().backward() # test backward propagation

    def test_model_snapshot_resume(self):
        model_store = get_model("unet", inchannels=12)
        filename = "/tmp/unet-dummy-model.pth"
        optimizer_store = torch.optim.Adam(model_store.parameters())
        epoch_store = 3
        logs_store = ["a","b"]
        snapshot(filename, model_store, optimizer_store, epoch_store, logs_store)

        # new blank model and optimizer
        model = get_model("unet", inchannels=12)
        optimizer = torch.optim.Adam(model.parameters())

        X = torch.ones(3, 12, 256, 256)
        y_pred_resumed = model(X)
        y_pred_stored = model_store(X)

        self.assertFalse((y_pred_stored == y_pred_resumed).all(), msg="predictions of stored and resumed models "
                                                                     "should be different before resuming weights."
                                                                      "If true, something in this test is wrong")

        epoch_resumed, logs_resumed = resume(filename, model, optimizer)

        self.assertTrue(logs_resumed == logs_store)
        self.assertTrue(epoch_resumed == epoch_store)

        X = torch.ones(3, 12, 256, 256)
        y_pred_resumed = model(X)
        y_pred_stored = model_store(X)

        self.assertTrue((y_pred_stored == y_pred_resumed).all(), msg="predictions of stored and resumed models "
                                                                     "should be identical")
from transforms import calculate_fdi, calculate_ndvi, get_transform, random_crop
from data import l2abands as bands

class TransformsTestCases(unittest.TestCase):

    def test_ndvi(self):

        for size in [
            (len(bands)),
            (len(bands), 10),
            (len(bands), 10, 20),
            (len(bands), 10, 20, 30)
        ]:

            with self.subTest(size=size):

                scene = np.ones(size)

                # set red to 0.2 and nir to 0.8. should result in ndvi = 0.6
                scene[bands.index("B4")] *= 0.2
                scene[bands.index("B8")] *= 0.8

                ndvi = calculate_ndvi(scene)

                self.assertAlmostEqual(ndvi.mean(), 0.6)


    def test_fdi(self):

        for size in [
            (len(bands)),
            (len(bands), 10),
            (len(bands), 10, 20),
            (len(bands), 10, 20, 30)
        ]:

            with self.subTest(size=size):

                scene = np.ones(size)


                scene[bands.index("B6")] *= 1682
                scene[bands.index("B8")] *= 1816
                scene[bands.index("B11")] *= 232

                fdi = calculate_fdi(scene)

                self.assertAlmostEqual(fdi.mean(), 0.2707, 4)


from create_overview_table import process_s2tiff, process_prediction
class COGconvertTestCases(unittest.TestCase):

    def test_convert_s2scene_to_viz(self):
        count = 12
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

        # tiff = "/ssd2/floatingObjects/data/accra_20181031.tif"
        process_s2tiff(tiff, upload=False, convert=False)

    def test_convert_prediction_to_viz(self):
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
        process_prediction(tiff, upload=False, convert=False)


class PredictTestCases(unittest.TestCase):

    def test_PythonPredictor(self):

        for width, height in product([32, 64, 128], [50, 100, 177]):

            with self.subTest(width=width, height=height):

                count = 12
                incrs = rasterio.crs.CRS.from_epsg(32636)
                intransform = rasterio.transform.Affine(10.0, 0.0, 502590.0,
                                                               0.0, -10.0, 3837650.0)

                imgmeta = {'driver': 'GTiff',
                        'dtype': 'uint16',
                        'nodata': None,
                        'width': width,
                        'height': height,
                        'count': count,
                        'crs': incrs,
                        'transform': intransform}

                imagepath = create_dummy_image(imgmeta)
                model = get_model("unet")
                snapshotpath = "/tmp/unet-dummy-model-snapshot.pth"
                torch.save(dict(model_state_dict=model.state_dict()), snapshotpath)

                predictor = PythonPredictor(snapshotpath, device="cpu")
                prediction_path = "/tmp/dummy_predictions"
                predictor.predict(imagepath, prediction_path)

                predicted_image = os.path.join(prediction_path, os.path.basename(imagepath))

                # predicted image has been created
                self.assertTrue(os.path.exists(predicted_image))

                with rasterio.open(predicted_image, "r") as src:
                    arr = src.read()
                    meta = src.meta

                # check if dimensions of predicted image are identical with input image
                self.assertTrue(meta["dtype"] == "uint8")
                self.assertTrue(arr.shape == (1, height, width))

                # check if prediction georeference is identical with input image
                self.assertTrue(incrs == meta["crs"])
                self.assertTrue(intransform == meta["transform"])

if __name__ == '__main__':
    unittest.main()
