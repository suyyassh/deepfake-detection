# this script creates manifests

import os
import pandas as pd
import random
import glob
from utils.config_loader import load_config, validate_config

# canonical manipulation-method names, used consistently across train and test.
# the train split encodes the full names in its filenames, while the test split
# uses abbreviated FF- tags; this map collapses both onto one vocabulary so the
# per-method evaluation table lines up. The map is idempotent: a canonical name
# maps to itself, so re-normalising is safe.
METHOD_MAP = {
    # canonical -> canonical (train-side names)
    'Deepfakes': 'Deepfakes',
    'Face2Face': 'Face2Face',
    'FaceSwap': 'FaceSwap',
    'FaceShifter': 'FaceShifter',
    'NeuralTextures': 'NeuralTextures',
    # FF- tags (test-side names) -> canonical
    'FF-DF': 'Deepfakes',
    'FF-F2F': 'Face2Face',
    'FF-FS': 'FaceSwap',
    'FF-FaceShifter': 'FaceShifter',
    'FF-NT': 'NeuralTextures',
}

def normalize_method(raw_tag):
    """
    maps a raw method tag (either a canonical name or an FF- test tag) to the
    single canonical method name. Raises KeyError on any unrecognised tag so a
    naming scheme we have not accounted for stops manifest generation loudly,
    rather than silently writing a wrong or fragmented per-method label.
    """
    if raw_tag not in METHOD_MAP:
        raise KeyError(
            f"Unrecognised method tag '{raw_tag}'. Known tags: {sorted(METHOD_MAP)}. "
            f"Add it to METHOD_MAP in create_manifests.py before continuing."
        )
    return METHOD_MAP[raw_tag]

def get_flattened_files(directory):
    """
    returns a list of flattened filenames found in a directory.
    """
    return [os.path.basename(f) for f in glob.glob(os.path.join(directory, '*.[jp][pn]*g'))]

def get_real_name(fake_name):
    """
    derives the exact real target filename from a fake filename.
    fake: train_Deepfakes_000_003_012.jpg --> Real: train_000_012.jpg
    """
    parts = fake_name.split('_')
    split_name = parts[0]
    target_id = parts[2]
    frame = parts[-1]
    return f"{split_name}_{target_id}_{frame}"

def generate_manifests(config):
    """
    generates training and testing manifests for the models.
    """
    dataset = config['data']['dataset']
    manip_dir = os.path.join('data', 'manipulated', dataset)
    raw_flat_dir = os.path.join('data', 'raw_flattened', dataset)
    manif_dir = os.path.join('data', 'manifests', dataset)

    # creating directories for baseline, novel and test
    for folder in ['baseline', 'novel', 'test']:
        os.makedirs(os.path.join(manif_dir, folder), exist_ok=True)

    # get all flattened filenames
    print("Update: Reading flattened files...")
    all_reals = get_flattened_files(os.path.join(raw_flat_dir, 'real'))
    all_fakes = get_flattened_files(os.path.join(raw_flat_dir, 'fake'))
    
    # fast lookup set
    all_reals_set = set(all_reals)

    # creating the train/val/test splits based on filename prefixes
    test_f = [f for f in all_fakes if f.startswith('test_')]
    train_val_f = [f for f in all_fakes if f.startswith('train_')]
    
    test_r = [f for f in all_reals if f.startswith('test_')]
    train_val_r = [f for f in all_reals if f.startswith('train_')]

    # shuffle and split train_val into 85/15 train/val split
    random.seed(42)
    random.shuffle(train_val_f)
    random.shuffle(train_val_r)
    
    val_f_end = int(len(train_val_f) * 0.15)
    val_r_end = int(len(train_val_r) * 0.15)
    
    val_f, train_f = train_val_f[:val_f_end], train_val_f[val_f_end:]
    val_r, train_r = train_val_r[:val_r_end], train_val_r[val_r_end:]

    def build_novel_manifest(f_list, out_name):
        """
        builds paired quadruplets for contrastive learning.
        """
        data = []
        for f_name in f_list:
            r_name = get_real_name(f_name)
            # ensure the matching real frame exists
            if r_name not in all_reals_set:
                continue
                
            data.append({
                'fake_fpr': os.path.join(manip_dir, 'fake_fpr', f_name),
                'fake_fpr_comp': os.path.join(manip_dir, 'fake_fpr_compressed', f_name),
                'real_raw': os.path.join(raw_flat_dir, 'real', r_name),
                'real_comp': os.path.join(manip_dir, 'real_compressed', r_name)
            })
        pd.DataFrame(data).to_csv(os.path.join(manif_dir, 'novel', out_name), index=False)
        
    def build_baseline_manifest(f_list, r_list, out_name):
        """
        builds image pairs of real/fake images for the baseline model.
        """
        data = []
        for f_name in f_list: 
            data.append({'path': os.path.join(raw_flat_dir, 'fake', f_name), 'label': 1})
        for r_name in r_list: 
            data.append({'path': os.path.join(raw_flat_dir, 'real', r_name), 'label': 0})
        
        df = pd.DataFrame(data).sample(frac=1, random_state=42)
        df.to_csv(os.path.join(manif_dir, 'baseline', out_name), index=False)
    
    def build_test_manifest(f_list, r_list, fake_dir, real_dir, out_name, use_raw=False):
        """
        builds test manifests and includes the manipulation method for deeper evaluation.
        """
        data = []
        for f_name in f_list:
            raw_method = f_name.split('_')[1]
            method = normalize_method(raw_method)
            f_path = os.path.join(raw_flat_dir, 'fake', f_name) if use_raw else os.path.join(manip_dir, fake_dir, f_name)
            data.append({'path': f_path, 'label': 1, 'method': method})
            
        for r_name in r_list:
            r_path = os.path.join(raw_flat_dir, 'real', r_name) if use_raw else os.path.join(manip_dir, real_dir, r_name)
            data.append({'path': r_path, 'label': 0, 'method': 'Real'})
        
        df = pd.DataFrame(data).sample(frac=1, random_state=42)
        df.to_csv(os.path.join(manif_dir, 'test', out_name), index=False)

    print("Update: creating manifest for training the novel model")
    build_novel_manifest(train_f, 'train.csv')
    build_novel_manifest(val_f, 'val.csv')

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