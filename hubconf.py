import torch
import gdown
import torch.nn as nn
device = "cuda"


def get_model(modelname, inchannels=12, pretrained=True):
    if modelname == "unet":
        # initialize model (random weights)
        model = UNet(n_channels=inchannels,
                     n_classes=1,
                     bilinear=False)
    elif modelname in ["resnetunet", "resnetunetscse"]:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34" if "resnet" in modelname else "efficientnet-b7",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            decoder_attention_type="scse" if modelname == "resnetunetscse" else None,
            classes=1,
        )
        model.encoder.conv1 = torch.nn.Conv2d(inchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    elif modelname in ["manet"]:
        import segmentation_models_pytorch as smp
        model = smp.MAnet(
            encoder_name="resnet34",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
        model.encoder.conv1 = torch.nn.Conv2d(inchannels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        raise ValueError(f"model {modelname} not recognized")
    return model


def unet_seed0(**kwargs):
    """
    U-Net pre-trained with for 50 epochs, batch size of 160, learning rate of 0.001, seed 0
    """
    add_fdi_ndvi = False
    no_pretrained = False
    url = 'https://drive.google.com/uc?export=download&id=1uZkaj7MPubCqCzSTTYS_57vbxKpgbOig'
    output = 'unet-posweight1-lr001-bs160-ep50-aug1-seed0.pth.tar'
    gdown.download(url, output, quiet=True)

    snapshot_file = torch.load(output)
    inchannels = 12 if not add_fdi_ndvi else 14
    model = get_model("unet", inchannels=inchannels, pretrained=not no_pretrained).to(device)
    model.load_state_dict(snapshot_file["model_state_dict"])
    return model


def unet_seed1(**kwargs):
    """
    U-Net pre-trained with for 50 epochs, batch size of 160, learning rate of 0.001, seed 1
    """
    add_fdi_ndvi = False
    no_pretrained = False
    url = 'https://drive.google.com/uc?export=download&id=1NRBv8W537fHaMVK6aR49MXpswQ_Fnnqj'
    output = 'unet-posweight1-lr001-bs160-ep50-aug1-seed1.pth.tar'
    gdown.download(url, output, quiet=True)

    snapshot_file = torch.load(output)
    inchannels = 12 if not add_fdi_ndvi else 14
    model = get_model("unet", inchannels=inchannels, pretrained=not no_pretrained).to(device)
    model.load_state_dict(snapshot_file["model_state_dict"])
    return model