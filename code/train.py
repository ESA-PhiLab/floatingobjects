import os
from tqdm import tqdm
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model import get_model
from data import FloatingSeaObjectDataset
from visualization import plot_batch
from transforms import get_transform
import json
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/data")
    parser.add_argument('--snapshot-path', type=str)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0, help="random seed for train/test region split")
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--augmentation-intensity', type=int, default=1, help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)")
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--add-fdi-ndvi', action="store_true")
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--no-pretrained', action="store_true")
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--tensorboard-logdir', type=str, default=None)
    parser.add_argument('--pos-weight', type=float, default=1, help="positional weight for the floating object class, large values counteract")
    args = parser.parse_args()
    # args.image_size = (args.image_size,args.image_size)

    return args


def main(args):
    data_path = args.data_path
    snapshot_path = args.snapshot_path

    batch_size = args.batch_size
    workers = args.workers
    image_size = args.image_size
    device = args.device
    n_epochs = args.epochs
    learning_rate = args.learning_rate

    tensorboard_logdir = args.tensorboard_logdir

    dataset = FloatingSeaObjectDataset(data_path, fold="train", transform=get_transform("train", intensity=args.augmentation_intensity, add_fdi_ndvi=args.add_fdi_ndvi),
                                       output_size=image_size, seed=args.seed)
    valid_dataset = FloatingSeaObjectDataset(data_path, fold="val", transform=get_transform("test", add_fdi_ndvi=args.add_fdi_ndvi),
                                             output_size=image_size, seed=args.seed, hard_negative_mining=False)

    # store run arguments in the same folder
    run_arguments = vars(args)
    run_arguments["train_regions"] = ", ".join(dataset.regions)
    run_arguments["valid_dataset"] = ", ".join(valid_dataset.regions)
    os.makedirs(os.path.dirname(args.snapshot_path), exist_ok=True)
    with open(os.path.join(os.path.dirname(args.snapshot_path), f"run_arguments_{args.seed}.json"), 'w') as outfile:
        json.dump(run_arguments, outfile)

    print(run_arguments)

    # loading training datasets
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=workers, shuffle=True)

    # compute the number of labels in each class
    # weights = compute_class_occurences(train_loader) #function that computes the occurences of the classes
    pos_weight = torch.FloatTensor([float(args.pos_weight)]).to(device)

    bcecriterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    def criterion(y_pred, target, mask=None):
        """a wrapper around BCEWithLogitsLoss that ignores no-data
        mask provides a boolean mask on valid data"""
        loss = bcecriterion(y_pred, target)
        if mask is not None:
            return (loss * mask.double()).mean()
        else:
            return loss.mean()

    inchannels = 12 if not args.add_fdi_ndvi else 14
    model = get_model(args.model, inchannels=inchannels, pretrained=not args.no_pretrained).to(device)



    # initialize optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if snapshot_path is not None and os.path.exists(snapshot_path):
        start_epoch, logs = resume(snapshot_path, model, optimizer)
        start_epoch += 1
        print(f"resuming from snapshot {snapshot_path}. starting epoch {start_epoch}")
        for log in logs:
            print(
                f"epoch {log['epoch']}: trainloss {log['trainloss']:.4f}, valloss {log['valloss']:.4f}. (from {snapshot_path})")
    else:
        start_epoch = 1
        logs = []

    # create summary writer if tensorboard_logdir is not None
    writer = SummaryWriter(log_dir=tensorboard_logdir) if tensorboard_logdir is not None else None

    for epoch in range(start_epoch, n_epochs + 1):
        trainloss = training_epoch(model, train_loader, optimizer, criterion, device)
        valloss, metrics = validating_epoch(model, val_loader, criterion, device)

        log = dict(
            epoch=epoch,
            trainloss=trainloss,
            valloss=valloss,
        )
        log.update(metrics)

        logs.append(log)

        if writer is not None:
            writer.add_scalars("loss", {"train": trainloss, "val": valloss}, global_step=epoch)
            fig = predict_images(val_loader, model, device)
            writer.add_figure("predictions", fig, global_step=epoch)

            predictions, targets = get_scores(val_loader, model, device)
            targets = targets.reshape(-1)
            targets = targets > 0.5 # make to bool
            predictions = predictions.reshape(-1)
            writer.add_pr_curve("unbalanced", targets, predictions, global_step=epoch)

            # make predictions and targets balanced by removing not floating pixels until numbers of positive
            # and negative samples are equal
            floating_predictions = predictions[targets]
            not_floating_predictions = predictions[~targets]
            np.random.shuffle(not_floating_predictions)
            not_floating_predictions = not_floating_predictions[:len(floating_predictions)]
            predictions = np.hstack([floating_predictions, not_floating_predictions])
            targets = np.hstack([np.ones_like(floating_predictions), np.zeros_like(not_floating_predictions)])
            writer.add_pr_curve("balanced", targets, predictions, global_step=epoch)

        # retrieve best loss by iterating through previous logged losses
        best_loss = min([l["valloss"] for l in logs])
        best_kappa = max([l["kappa"] for l in logs])
        kappa = metrics["kappa"]

        save_msg = ""  # write save model message in the same line of the pring
        if valloss <= best_loss or kappa >= best_kappa:
            save_msg = f"saving model to {snapshot_path}"  # add this message if model saved
            snapshot(snapshot_path, model, optimizer, epoch, logs)

        metrics_message = ", ".join([f"{k} {v:.2f}" for k,v in metrics.items()])

        print(f"epoch {epoch}: trainloss {trainloss:.4f}, valloss {valloss:.4f}, {metrics_message} ,{save_msg}")


def training_epoch(model, train_loader, optimizer, criterion, device):
    losses = []
    model.train()
    with tqdm(enumerate(train_loader), total=len(train_loader), leave=False) as pbar:
        for idx, batch in pbar:
            optimizer.zero_grad()
            im, target, id = batch
            im = im.to(device)
            target = target.to(device)
            y_pred = model(im)
            valid_data = im.sum(1) != 0 # all pixels > 0
            loss = criterion(y_pred.squeeze(1), target, mask=valid_data)
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            pbar.set_description(f'train loss {np.array(losses).mean():.4f}')
    return np.array(losses).mean()


def validating_epoch(model, val_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        losses = []
        metrics = dict(
            precision = [],
            recall = [],
            fscore = [],
            kappa = []
        )
        with tqdm(enumerate(val_loader), total=len(val_loader), leave=False) as pbar:
            for idx, batch in pbar:
                im, target, id = batch
                im = im.to(device)
                target = target.to(device)
                y_pred = model(im)
                valid_data = im.sum(1) != 0  # all pixels > 0
                loss = criterion(y_pred.squeeze(1), target, mask=valid_data)
                losses.append(loss.cpu().detach().numpy())
                pbar.set_description(f'val loss {np.array(losses).mean():.4f}')
                predictions = (y_pred.exp() > 0.5).cpu().detach().numpy()
                y_true = target.cpu().view(-1).numpy().astype(bool)
                y_pred = predictions.reshape(-1)
                p,r,f,s = precision_recall_fscore_support(y_true=y_true,
                                                y_pred=y_pred, zero_division=0)
                metrics["kappa"] = cohen_kappa_score(y_true, y_pred)
                metrics["precision"].append(p)
                metrics["recall"].append(r)
                metrics["fscore"].append(f)

    for k,v in metrics.items():
        metrics[k] = np.array(v).mean()

    return np.array(losses).mean(), metrics


def predict_images(val_loader, model, device):
    images, masks, id = next(iter(val_loader))
    N = images.shape[0]

    # plot at most 5 images even if the batch size is larger
    if N > 5:
        images = images[:5]
        masks = masks[:5]

    logits = model(images.to(device)).squeeze(1)
    y_preds = torch.sigmoid(logits).detach().cpu().numpy()
    return plot_batch(images, masks, y_preds)


def get_scores(val_loader, model, device, n_batches=5):
    y_preds = []
    targets = []
    with torch.no_grad():
        for i in range(n_batches):
            images, masks, id = next(iter(val_loader))
            logits = model(images.to(device)).squeeze(1)
            y_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets.append(masks.detach().cpu().numpy())
    return np.vstack(y_preds), np.vstack(targets)

def snapshot(filename, model, optimizer, epoch, logs):
    torch.save(dict(model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    epoch=epoch,
                    logs=logs),
               filename)


def resume(filename, model, optimizer):
    snapshot_file = torch.load(filename)
    model.load_state_dict(snapshot_file["model_state_dict"])
    optimizer.load_state_dict(snapshot_file["optimizer_state_dict"])
    return snapshot_file["epoch"], snapshot_file["logs"]


def compute_class_occurences(train_loader):
    sum_no_floating, sum_floating = 0, 0
    for idx, (_, target, _) in enumerate(train_loader):
        sum_no_floating += torch.sum(target == 0)
        sum_floating += torch.sum(target == 1)
    return sum_no_floating, sum_floating


if __name__ == '__main__':
    args = parse_args()
    '''
    if os.environ['HOME'] == '/home/jmifdal':
        args.data_path = os.environ['HOME'] + "/data/floatingobjects/"
        args.epochs = 50
        args.image_size = 128
        args.snapshot_path = os.environ['HOME'] + '/remote/floatingobjects/models/model_ratio10_22_01_2021.pth.tar'
        args.tensorboard_logdir = os.environ['HOME'] + '/remote/floatingobjects/models/runs/run_ratio10_22_01_2021/'
    '''
    main(args)

# --data-path /home/jmifdal/data/floatingobjects/ --snapshot-path /home/jmifdal/remote/floatingobjects/models/model_18_01_2021.pth.tar --image-size 128 --epoch 50 --tensorboard-logdir /home/jmifdal/remote/floatingobjects/models/runs/run_18_01_2021/

'''
train.py --data-path /home/jmifdal/data/floatingobjects/ --snapshot-path /home/jmifdal/remote/floatingobjects/models/model.pth.tar --image-size (128,128) --epochs 50 --tensorboard-logdir /home/jmifdal/remote/floatingobjects/models/runs/
'''
