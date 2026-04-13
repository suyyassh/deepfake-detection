# this script creates manifests

import os
import pandas as pd
import random
import glob
from utils.config_loader import load_config, validate_config

def get_flattened_files(directory):
    """
    return a list of flattened filenames found in a directory
    """
    # get all the paths where the file ends with .jpg or .png and then strip the entire path
    # and reutrn just the filename i.e. /data/images/face01.jpg becomes face01.jpg
    return [os.path.basename(f) for f in glob.glob(os.path.join(directory, '*.[jp][pn]*g'))]

def build_raw_path_map(raw_dir, category):
    """
    creates a dictionary that maps a flattened key to a path.
    we need flattened keys as they will act as "keys" for looking up manipulated
    images when creating manifests.
    """
    path_map = {}

    # raw_dir / real or fake / <any subdirectories, however deep> / <any image file>
    search_pattern = os.path.join(raw_dir, category, '**', '*.[jp][pn]*g')

    # for all the files that match the search pattern, create a flattened key (foldername_filename)
    # and map that key to the path
    for path in glob.glob(search_pattern, recursive=True):
        fname = f"{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)}"
        path_map[fname] = path
    
    return path_map

def split_filenames(filenames, train_pct=0.7, val_pct=0.15):
    """
    splits filenames into 70/15/15 while maintaining identity isolation
    """
    random.seed(42)
    random.shuffle(filenames) 
    n = len(filenames)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)
    return filenames[:train_end], filenames[train_end:val_end], filenames[val_end:]

def generate_manifests(config):
    """
    does what it says it does
    """
    # setting up paths
    dataset = config['data']['dataset']
    raw_dir = config['data']['raw_dir']
    manip_dir = os.path.join('data', 'manipulated', dataset)
    manif_dir = os.path.join('data', 'manifests', dataset)

    # creating directories for baseline, novel and test
    for folder in ['baseline', 'novel', 'test']:
        os.makedirs(os.path.join(manif_dir, folder), exist_ok=True)

    # mapping flattened keys for RAW files to their paths
    print("Update: mapping flattened keys for RAW images (real and fake) to their paths")
    raw_real_map = build_raw_path_map(raw_dir, 'real') 
    raw_fake_map = build_raw_path_map(raw_dir, 'fake')

    # getting flattened filenames real compressed and fingerprint removed fakes
    all_reals = get_flattened_files(os.path.join(manip_dir, 'real_compressed'))
    all_fakes = get_flattened_files(os.path.join(manip_dir, 'fake_fpr'))

    # create the train/validate/test split
    train_r, val_r, test_r = split_filenames(all_reals)
    train_f, val_f, test_f = split_filenames(all_fakes)
 
    def build_novel_manifest(f_list, r_list, out_name):
        """
        builds a four image quadruplets for contrastive learning
        we will not apply labels for this on the fly when feeding data to the model
        """
        data = []
        for f_name in f_list:
            r_name = random.choice(r_list) # pairs a fake with a random real identity
            data.append({
                'fake_fpr': os.path.join(manip_dir, 'fake_fpr', f_name), # fingerprint removed fake
                'fake_fpr_comp': os.path.join(manip_dir, 'fake_fpr_compressed', f_name), # fingerprint removed compressed fake
                'real_raw': raw_real_map[r_name], # real RAW image
                'real_comp': os.path.join(manip_dir, 'real_compressed', r_name) # compressed real
            })
        pd.DataFrame(data).to_csv(os.path.join(manif_dir, 'novel', out_name), index=False)
        
    def build_baseline_manifest(f_list, r_list, out_name):
        """
        builds image pairs of real/fake images for training the baseline model
        """
        data = []
        for f_name in f_list: data.append({'path': raw_fake_map[f_name], 'label':1})
        for r_name in r_list: data.append({'path': raw_real_map[r_name], 'label':0})
        
        # shuffle the rows so that model doesn't see all fakes and then all reals
        df = pd.DataFrame(data).sample(frac=1, random_state=42)
        df.to_csv(os.path.join(manif_dir, 'baseline', out_name), index=False)
    
    def build_test_manifest(f_list, r_list, fake_dir, real_dir, out_name, use_raw=False):
        """
        build image pairs for testing the real/fake images for testing both models
        """
        data = []
        for f_name in f_list:
            f_path = raw_fake_map[f_name] if use_raw else os.path.join(manip_dir, fake_dir, f_name)
            data.append({'path': f_path, 'label': 1})
        for r_name in r_list:
            r_path = raw_real_map[r_name] if use_raw else os.path.join(manip_dir, real_dir, r_name)
            data.append({'path': r_path, 'label': 0})
        
        df = pd.DataFrame(data).sample(frac=1, random_state=42)
        df.to_csv(os.path.join(manif_dir, 'test', out_name), index=False)

    print("Update: creating manifest for training the novel model")
    build_novel_manifest(train_f, train_r, 'train.csv')
    build_novel_manifest(val_f, val_r, 'val.csv')

    print("Update: creating manifest for training the baseline model")
    build_baseline_manifest(train_f, train_r, 'train.csv')
    build_baseline_manifest(val_f, val_r, 'val.csv')

    print("Update: creating manifest for testing both models")
    build_test_manifest(test_f, test_r, None, None, 'test_raw.csv', use_raw=True)
    build_test_manifest(test_f, test_r, 'fake_compressed', 'real_compressed', 'test_compressed.csv', use_raw=False)

    print(f"Success: all manifests saved to {manif_dir}")

if __name__ == "__main__":
    cfg = load_config("configs/base_config.yaml")
    validate_config(cfg)
    generate_manifests(cfg)