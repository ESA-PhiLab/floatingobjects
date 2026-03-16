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

    # Ensure the directory exists
    print(f"Creating {os.path.abspath(args.storepath)}")
    os.makedirs(args.storepath, exist_ok=True)

    # Specify the path including the filename where the file will be saved
    zip_path = os.path.join(args.storepath, "data.zip")

    # Download
    print(f"Downloading {URL} to {zip_path}")
    gdown.download(URL, zip_path, quiet=False)

    # Unpack
    print(f"Unpacking {zip_path} to {os.path.abspath(args.storepath)}")
    shutil.unpack_archive(zip_path, os.path.abspath(args.storepath))

    # Optionally, you can delete the zip file after unpacking if it's no longer needed
    # os.remove(zip_path)
