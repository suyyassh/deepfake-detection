# this script creates subsets like compressed real images or fingerprint removed fakes etc.

import os
import cv2
import glob # library for file search
from tqdm import tqdm # library to show progress bar
from utils.config_loader import load_config, validate_config
from preprocessing.helpers import remove_noise, apply_compression

def process_data(config):
    """
    this function does _____________
    """
    raw_dir = config['data']['raw_dir'] 
    dataset_name = config['data']['dataset']

    # creating directories for manipulated content
    manip_root = os.path.join('data', 'manipulated', dataset_name)
    folders = {
        'real_c': os.path.join(manip_root, 'real_compressed'),
        'fake_c': os.path.join(manip_root, 'fake_compressed'),
        'fake_fpr': os.path.join(manip_root, 'fake_fpr'),
        'fake_fpr_c': os.path.join(manip_root, 'fake_fpr_compressed')
    }

    for path in folders.values():
        os.makedirs(path, exist_ok=True)

    # gather RAW file from the config
    # raw_dir / real or fake / <any subdirectories, however deep> / <any image file>
    raw_reals = glob.glob(os.path.join(raw_dir, 'real', '**', '*.[jp][pn]*g'), recursive=True)
    raw_fakes = glob.glob(os.path.join(raw_dir, 'fake', '**', '*.[jp][pn]*g'), recursive=True)

    print(f"Update: applying custom manipulations for {dataset_name}")

    # manipulating RAW real images
    for path in tqdm(raw_reals, desc="Reals"):
        img = cv2.imread(path)
        if img is None: continue

        # naming files such as 0000_0000
        fname = f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}"

        # compressing RAW reals
        cv2.imwrite(os.path.join(folders['real_c'], fname), apply_compression(img))

    # manipulating RAW fake images
    for path in tqdm(raw_fakes, desc="Fakes"):
        img = cv2.imread(path)
        if img is None: continue

        # naming files such as 0000_fake_000
        fname = f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}"

        # compressing RAW fakes
        img_comp = apply_compression(img)
        cv2.imwrite(os.path.join(folders['fake_c'], fname), img_comp)

        # removing fingerprints from RAW fakes
        img_fpr = remove_noise(img)
        cv2.imwrite(os.path.join(folders['fake_fpr'], fname), img_fpr)

        # compressing fingerprint removed fakes
        img_fpr_comp = apply_compression(img_fpr)
        cv2.imwrite(os.path.join(folders['fake_fpr_c'], fname), img_fpr_comp)

if __name__ == "__main__":
    cfg = load_config("configs/base_config.yaml")
    validate_config(cfg)
    process_data(cfg)