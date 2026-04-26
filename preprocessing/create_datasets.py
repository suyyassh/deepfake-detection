# this script creates subsets like compressed real images or fingerprint removed fakes etc.

import os
import cv2
import glob # library for file search
from tqdm import tqdm # library to show progress bar
from utils.config_loader import load_config, validate_config
from preprocessing.helpers import remove_noise, apply_compression

def process_data(config):
    """
    applies custom manipulations to the dataset and flattens the directory structure
    for easier manifest generation.
    """
    raw_dir = config['data']['raw_dir'] 
    dataset_name = config['data']['dataset']

    # creating directories for manipulated content and flattened raw content
    manip_root = os.path.join('data', 'manipulated', dataset_name)
    raw_flat_root = os.path.join('data', 'raw_flattened', dataset_name) 
    
    folders = {
        'real_raw': os.path.join(raw_flat_root, 'real'),
        'fake_raw': os.path.join(raw_flat_root, 'fake'),
        'real_c': os.path.join(manip_root, 'real_compressed'),
        'fake_c': os.path.join(manip_root, 'fake_compressed'),
        'fake_fpr': os.path.join(manip_root, 'fake_fpr'),
        'fake_fpr_c': os.path.join(manip_root, 'fake_fpr_compressed')
    }

    for path in folders.values():
        os.makedirs(path, exist_ok=True)

    print(f"Update: gathering RAW files for {dataset_name}")

    # gathering RAW files and accounting for 'train' vs 'test' naming differences
    splits = ['train', 'test']
    raw_reals = []
    raw_fakes = []
    
    for split in splits:
        # if split == test, fake_dir == 'FF-fake', else fake, same for real_dir
        fake_dir = 'FF-fake' if split == 'test' else 'fake'
        real_dir = 'FF-real' if split == 'test' else 'real'
        
        # raw_dir/<train or test>/<FF-fake/real or fake/real>/<any_depth>/<end with .jpg, .jpeg, .png> and
        # add the file paths to raw_reals and raw_fakes
        raw_reals.extend(glob.glob(os.path.join(raw_dir, split, real_dir, '**', '*.[jp][pn]*g'), recursive=True))
        raw_fakes.extend(glob.glob(os.path.join(raw_dir, split, fake_dir, '**', '*.[jp][pn]*g'), recursive=True))

    print(f"Update: applying custom manipulations for {dataset_name}")

    # manipulating RAW real images
    for path in tqdm(raw_reals, desc="Reals"):
        img = cv2.imread(path)
        if img is None: continue

        # extract path components to build {split}_{target}_{frame}
        parts = os.path.normpath(path).split(os.sep)
        split_name = parts[-4] # train or test
        target_id = parts[-2] # 000
        frame = parts[-1] # 000.png
        
        fname = f"{split_name}_{target_id}_{frame}"

        # save flattened raw and compressed
        cv2.imwrite(os.path.join(folders['real_raw'], fname), img)
        cv2.imwrite(os.path.join(folders['real_c'], fname), apply_compression(img))

    # manipulating RAW fake images
    for path in tqdm(raw_fakes, desc="Fakes"):
        img = cv2.imread(path)
        if img is None: continue

        # extract path components to build: {split}_{method}_{pair}_{frame}
        parts = os.path.normpath(path).split(os.sep)
        split_name = parts[-5] # train or test
        method = parts[-3] # Deepfakes, FF-DF etc.
        pair_id = parts[-2] # 000_003
        frame = parts[-1] # 012.png

        fname = f"{split_name}_{method}_{pair_id}_{frame}"

        # save flattened raw
        cv2.imwrite(os.path.join(folders['fake_raw'], fname), img)
        
        # save manipulations
        img_comp = apply_compression(img)
        cv2.imwrite(os.path.join(folders['fake_c'], fname), img_comp)

        img_fpr = remove_noise(img)
        cv2.imwrite(os.path.join(folders['fake_fpr'], fname), img_fpr)

        img_fpr_comp = apply_compression(img_fpr)
        cv2.imwrite(os.path.join(folders['fake_fpr_c'], fname), img_fpr_comp)

if __name__ == "__main__":
    cfg = load_config("configs/base_config.yaml")
    validate_config(cfg)
    process_data(cfg)