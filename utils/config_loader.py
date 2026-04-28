## not updated in this specific implementation

# this script loads and validates the configuration

import yaml
import os

def load_config(config_path):
    """
    reads a yaml file and converts it into a nested python dictionary
    """
    # if the file is not found at the desired path raise an error
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Error: config file not found at {config_path}")
    
    # if the file is found, open the file in read mode
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    return config


def validate_config(config):
    """
    Ensures no hyper-parameter is empty and types are correct
    """
    schema = [
        ('data', 'dataset', str),
        ('data', 'raw_dir', str),
        ('data', 'img_size', int),
        ('data', 'num_workers', int),
        ('model', 'backbone', str),
        ('model', 'pretrained', bool),
        ('model', 'embedding_dim', int),
        ('train', 'batch_size', int),
        ('train', 'epochs', int),
        ('train', 'learning_rate', float),
        ('train', 'weight_decay', float),
    ]

    for section, key, expected_type in schema:
        # check if section and key exist
        if section not in config or key not in config[section]:
            raise KeyError(f"Error: '{section}.{key}' is missing!")
        
        value = config[section][key]

        # check if the value is not none
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ValueError(f"Error: '{section}.{key}' cannot be empty")
        
        # check if the type is correct
        if not isinstance(value, expected_type):
            type_name = "numeric" if isinstance(expected_type, tuple) else expected_type.__name__
            raise TypeError(f"Error: '{section}.{key}' must be {type_name}")
        
if __name__  == "__main__":
    cfg = load_config("configs/base_config.yaml")
    try:
        validate_config(cfg)
        print(f"Success: {cfg['project_name']} now loaded")
        print(f"Backbone selected: {cfg['model']['backbone']}")
        print(f"Dataset selected: {cfg['data']['dataset']}")
    except (KeyError, ValueError, TypeError) as error:
        print(f"{error}")