import argparse
from train import main as train
import os

def main(args):
    if args.mode == "train":
        for seed in range(5):
            args.seed = seed
            args.snapshot_path = os.path.join(args.results_dir, f"model_{seed}.pth.tar")
            args.tensorboard_logdir = os.path.join(args.tensorboard, f"model_{seed}/")
            os.makedirs(args.results_dir, exist_ok=True)
            os.makedirs(args.tensorboard, exist_ok=True)
            train(args)
    elif args.mode == "predict":
        raise NotImplementedError("here all images should be predicted in a loop using the models trained before")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=["train", "predict"])
    parser.add_argument('--results-dir', type=str, default="/tmp/floatingobjects")
    parser.add_argument('--tensorboard', type=str, default=None)    

    # train arguments
    parser.add_argument('--data-path', type=str, default="/data")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--augmentation-intensity', type=int, default=1, help="number indicating intensity 0, 1 (noise), 2 (channel shuffle)")
    parser.add_argument('--model', type=str, default="unet")
    parser.add_argument('--add-fdi-ndvi', action="store_true")
    parser.add_argument('--no-pretrained', action="store_true")
    #parser.add_argument('--tensorboard-logdir', type=str, default=None)    
    parser.add_argument('--pos-weight', type=float, default=1, help="positional weight for the floating object class, large values counteract")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(parse_args())