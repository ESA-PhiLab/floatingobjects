from model import get_model
import torch
import rasterio
from rasterio.windows import Window
import numpy as np
from itertools import product
from data import l1cbands, l2abands
import os
from tqdm import tqdm
import argparse
from transforms import get_transform
from scipy.ndimage.filters import gaussian_filter, median_filter
from data import get_region_split

class PythonPredictor():

    #def __init__(self, modelname, modelpath, image_size=(480,480), device="cpu", offset=64, use_test_aug=2, add_fdi_ndvi=False):
    def __init__(self, modelpath, image_size=(480,480), device="cpu", offset=64, use_test_aug=2, add_fdi_ndvi=False):
        self.image_size = image_size
        self.device = device
        self.offset = offset # remove border effects from the CNN

        #self.model = UNet(n_channels=12, n_classes=1, bilinear=False)
        self.model = get_model(os.path.basename(modelpath).split("-")[0].lower(), inchannels=12 if not add_fdi_ndvi else 14)
        #self.model = get_model(modelname, inchannels=12 if not add_fdi_ndvi else 14)
        self.model.load_state_dict(torch.load(modelpath)["model_state_dict"])
        self.model = self.model.to(device)
        self.transform = get_transform("test", add_fdi_ndvi=add_fdi_ndvi)
        self.use_test_aug = use_test_aug

    def predict(self, path, predpath):
        with rasterio.open(path, "r") as src:
            meta = src.meta
        self.model.eval()

        # for prediction
        predimage = os.path.join(predpath, os.path.basename(path))
        os.makedirs(predpath, exist_ok=True)
        meta["count"] = 1
        meta["dtype"] = "uint8" # storing as uint8 saves a lot of storage space

        #Window(col_off, row_off, width, height)
        H, W = self.image_size

        rows = np.arange(0, meta["height"], H)
        cols = np.arange(0, meta["width"], W)

        image_window = Window(0, 0, meta["width"], meta["height"])

        with rasterio.open(predimage, "w+", **meta) as dst:

            for r, c in tqdm(product(rows, cols), total=len(rows) * len(cols), leave=False):

                window = image_window.intersection(
                    Window(c-self.offset, r-self.offset, W+self.offset, H+self.offset))

                with rasterio.open(path) as src:
                    image = src.read(window=window)

                # if L1C image (13 bands). read only the 12 bands compatible with L2A data
                if (image.shape[0] == 13):
                    image = image[[l1cbands.index(b) for b in l2abands]]

                # to torch + normalize
                image = self.transform(torch.from_numpy(image.astype(np.float32)), [])[0].to(self.device)

                # predict
                with torch.no_grad():
                    x = image.unsqueeze(0)
                    #import pdb; pdb.set_trace()
                    y_logits = torch.sigmoid(self.model(x).squeeze(0))
                    if self.use_test_aug > 0:
                        y_logits += torch.sigmoid(torch.fliplr(self.model(torch.fliplr(x)))).squeeze(0) # fliplr)
                        y_logits += torch.sigmoid(torch.flipud(self.model(torch.flipud(x)))).squeeze(0) # flipud
                        if self.use_test_aug > 1:
                            for rot in [1, 2, 3]: # 90, 180, 270 degrees
                                y_logits += torch.sigmoid(torch.rot90(self.model(torch.rot90(x, rot, [2, 3])),-rot,[2,3]).squeeze(0))
                            y_logits /= 6
                        else:
                            y_logits /= 3

                    y_score = y_logits.cpu().detach().numpy()[0]
                    #y_score = y_score[:,self.offset:-self.offset, self.offset:-self.offset]

                data = dst.read(window=window)[0] / 255
                overlap = data > 0

                if overlap.any():
                    # smooth transition in overlapping regions
                    dx, dy = np.gradient(overlap.astype(float)) # get border
                    g = np.abs(dx) + np.abs(dy)
                    transition = gaussian_filter(g, sigma=self.offset / 2)
                    transition /= transition.max()
                    transition[~overlap] = 1.# normalize to 1

                    y_score = transition * y_score + (1-transition) * data

                # write
                writedata = (np.expand_dims(y_score, 0).astype(np.float32) * 255).astype(np.uint8)
                dst.write(writedata, window=window)

def main(args):
    predictor = PythonPredictor(args.snapshot_path, args.image_size, device="cuda", add_fdi_ndvi=args.add_fdi_ndvi)
    #predictor = PythonPredictor(args.model, args.snapshot_path, args.image_size, device="cuda", add_fdi_ndvi=args.add_fdi_ndvi)
    if args.image_path is not None:
        predictor.predict(args.image_path, args.prediction_path)
    else:
        regions_split = get_region_split(args.seed)
        valregions = regions_split["val"]
        testregions = regions_split["test"]
        os.listdir(args.image_folder)
        for region in valregions:
            l1cpath = os.path.join(args.image_folder, region + ".tif")
            l2apath = os.path.join(args.image_folder, region + "_l2a.tif")
            imagepath = l2apath if os.path.exists(l2apath) else l1cpath
            storepath = os.path.join(args.prediction_path, str(args.seed), "val")
            os.makedirs(storepath, exist_ok=True)
            print(f"predicting {imagepath} -> {storepath}")
            predictor.predict(imagepath, storepath)

        for region in testregions:
            l1cpath = os.path.join(args.image_folder, region + ".tif")
            l2apath = os.path.join(args.image_folder, region + "_l2a.tif")
            imagepath = l2apath if os.path.exists(l2apath) else l1cpath
            storepath = os.path.join(args.prediction_path, str(args.seed), "test")
            os.makedirs(storepath, exist_ok=True)
            print(f"predicting {imagepath} -> {storepath}")
            predictor.predict(imagepath, storepath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, default=None)
    parser.add_argument('--image-folder', type=str, default=None)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--prediction-path', type=str, default="/data")
    parser.add_argument('--snapshot-path', type=str)
    parser.add_argument('--add-fdi-ndvi', action="store_true")

    args = parser.parse_args()
    args.image_size = (args.image_size,args.image_size)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)