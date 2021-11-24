import torch
import gdown
import os
import sys
this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_folder, "code"))

from model import get_model
#device = "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unet_seed0(**kwargs):
    """
    U-Net pre-trained with for 50 epochs, with batch size of 160, learning rate of 0.001, seed 0
    """
    add_fdi_ndvi = False
    no_pretrained = False
    url = 'https://drive.google.com/uc?export=download&id=1uZkaj7MPubCqCzSTTYS_57vbxKpgbOig'
    output = 'unet-posweight1-lr001-bs160-ep50-aug1-seed0.pth.tar'
    gdown.download(url, output, quiet=True)

    snapshot_file = torch.load(output, map_location=device)
    inchannels = 12 if not add_fdi_ndvi else 14
    model = get_model("unet", inchannels=inchannels, pretrained=not no_pretrained).to(device)
    model.load_state_dict(snapshot_file["model_state_dict"])
    return model


def unet_seed1(**kwargs):
    """
    U-Net pre-trained with for 50 epochs, with batch size of 160, learning rate of 0.001, seed 1
    """
    add_fdi_ndvi = False
    no_pretrained = False
    url = 'https://drive.google.com/uc?export=download&id=1NRBv8W537fHaMVK6aR49MXpswQ_Fnnqj'
    output = 'unet-posweight1-lr001-bs160-ep50-aug1-seed1.pth.tar'
    gdown.download(url, output, quiet=True)

    snapshot_file = torch.load(output, map_location=device)
    inchannels = 12 if not add_fdi_ndvi else 14
    model = get_model("unet", inchannels=inchannels, pretrained=not no_pretrained).to(device)
    model.load_state_dict(snapshot_file["model_state_dict"])
    return model


def manet_seed0(**kwargs):
    """
    MA-Net pre-trained for 50 epochs, with batch size of 160, learning rate of 0.001, seed 0
    """
    add_fdi_ndvi = False
    no_pretrained = False
    url = 'https://drive.google.com/uc?export=download&id=1RWSS1AJweAgBJRftr6TCMb9vNN_hTNXf'
    output = 'manet-posweight1-lr001-bs160-ep50-aug1-seed0.pth.tar'
    gdown.download(url, output, quiet=True)

    snapshot_file = torch.load(output, map_location=device)
    inchannels = 12 if not add_fdi_ndvi else 14
    model = get_model("manet", inchannels=inchannels, pretrained=not no_pretrained).to(device)
    model.load_state_dict(snapshot_file["model_state_dict"])
    return model


def manet_seed1(**kwargs):
    """
    MA-Net pre-trained for 50 epochs, with batch size of 160, learning rate of 0.001, seed 1
    """
    add_fdi_ndvi = False
    no_pretrained = False
    url = 'https://drive.google.com/uc?export=download&id=17I2PJS947p71EV-zZhl6laPemknl2_UY'
    output = 'manet-posweight1-lr001-bs160-ep50-aug1-seed1.pth.tar'
    gdown.download(url, output, quiet=True)

    snapshot_file = torch.load(output, map_location=device)
    inchannels = 12 if not add_fdi_ndvi else 14
    model = get_model("manet", inchannels=inchannels, pretrained=not no_pretrained).to(device)
    model.load_state_dict(snapshot_file["model_state_dict"])
    return model
