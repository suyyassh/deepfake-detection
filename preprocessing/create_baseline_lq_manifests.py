# this script creates manifests for training the baseline model on compressed

import os
import pandas as pd
import random
import glob
from utils.config_loader import load_config, validate_config


def get_flattened_files(directory):
    """
    returns a list of flattened filenames found in a directory.
    """
    return [os.path.basename(f)
            for f in glob.glob(os.path.join(directory, '*.[jp][pn]*g'))]


def generate_lq_manifests(config):
    """
    generates train/val manifests for baseline training on compressed images.
    the file list and split are derived from the raw_flattened directory so
    that the train/val partition is identical to the standard baseline split —
    the only difference is that every image path points at the compressed copy.
    """
    dataset     = config['data']['dataset']
    manip_dir   = os.path.join('data', 'manipulated', dataset)
    raw_flat_dir = os.path.join('data', 'raw_flattened', dataset)
    manif_dir   = os.path.join('data', 'manifests', dataset)
    out_dir     = os.path.join(manif_dir, 'baseline_lq')

    os.makedirs(out_dir, exist_ok=True)
    print("Update: reading flattened file lists...")
    all_fakes = get_flattened_files(os.path.join(raw_flat_dir, 'fake'))
    all_reals = get_flattened_files(os.path.join(raw_flat_dir, 'real'))

    # keep only train-prefixed files; test frames are never used for training
    train_val_f = [f for f in all_fakes if f.startswith('train_')]
    train_val_r = [f for f in all_reals if f.startswith('train_')]

    # identical shuffle + split to create_manifests.py (seed 42, 85/15)
    random.seed(42)
    random.shuffle(train_val_f)
    random.shuffle(train_val_r)

    val_f_end = int(len(train_val_f) * 0.15)
    val_r_end = int(len(train_val_r) * 0.15)

    val_f,   train_f = train_val_f[:val_f_end], train_val_f[val_f_end:]
    val_r,   train_r = train_val_r[:val_r_end], train_val_r[val_r_end:]

    def build_lq_manifest(f_list, r_list, out_name):
        data = []
        for f_name in f_list:
            path = os.path.join(manip_dir, 'fake_compressed', f_name)
            data.append({'path': path, 'label': 1})
        for r_name in r_list:
            path = os.path.join(manip_dir, 'real_compressed', r_name)
            data.append({'path': path, 'label': 0})

        df = pd.DataFrame(data).sample(frac=1, random_state=42)
        out_path = os.path.join(out_dir, out_name)
        df.to_csv(out_path, index=False)
        print(f"Update: saved {out_path}  ({len(df)} rows — "
              f"{(df['label']==1).sum()} fake, {(df['label']==0).sum()} real)")

    print("Update: building baseline_lq train manifest...")
    build_lq_manifest(train_f, train_r, 'train.csv')

    print("Update: building baseline_lq val manifest...")
    build_lq_manifest(val_f, val_r, 'val.csv')

    print(f"Success: baseline_lq manifests saved to {out_dir}")


if __name__ == "__main__":
    cfg = load_config("configs/base_config.yaml")
    validate_config(cfg)
    generate_lq_manifests(cfg)