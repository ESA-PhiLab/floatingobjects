import gdown
import argparse
import os
import shutil

URL = 'https://drive.google.com/uc?id=10SGFhHMSnikgm9q90jJmHDVkhD_SZl7p'

if __name__ == "__main__":
    this_folder = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='A simple script to download the Dataset.')
    parser.add_argument('storepath', type=str, nargs='?', default=os.path.join(this_folder, "..", "data"))
    args = parser.parse_args()

    print(f"creating {os.path.abspath(args.storepath)}")
    os.makedirs(args.storepath, exist_ok=True)

    # download
    print(f"downloading {URL} to {os.path.abspath(args.storepath)}")
    gdown.download(URL, args.storepath, quiet=False)

    # unpack
    print(f"unpacking {args.storepath}/data.zip to {os.path.abspath(args.storepath)}")
    shutil.unpack_archive(os.path.join(args.storepath, "data.zip"), os.path.join(args.storepath, ".."))