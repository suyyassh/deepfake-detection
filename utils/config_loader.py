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
    Ensures no hyper-parameter is empty and types are correct.

    Notes:
    - learning_rate accepts int or float. PyYAML parses an unquoted "1e-4" as a
      STRING, which would otherwise pass silently and break the optimiser, so we
      coerce numeric-looking strings to float here and write the value back.
    - positive-value checks guard the sizes that must be >= 1.
    """
    # (section, key, expected_type, must_be_positive)
    schema = [
        ('data', 'dataset', str, False),
        ('data', 'raw_dir', str, False),
        ('data', 'img_size', int, True),
        ('data', 'num_workers', int, False),
        ('data', 'train_fraction', (int, float), True),
        ('model', 'backbone', str, False),
        ('model', 'pretrained', bool, False),
        ('model', 'embedding_dim', int, True),
        ('train', 'batch_size_base', int, True),
        ('train', 'batch_size_novel', int, True),
        ('train', 'epochs', int, True),
        ('train', 'warmup_epochs', int, False),
        ('train', 'learning_rate', (int, float), True),
    ]

    for section, key, expected_type, must_be_positive in schema:
        # check if section and key exist
        if section not in config or key not in config[section]:
            raise KeyError(f"Error: '{section}.{key}' is missing!")

        value = config[section][key]

        # check if the value is not none / not an empty string
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ValueError(f"Error: '{section}.{key}' cannot be empty")

        # special handling for learning_rate: rescue scientific notation that
        # PyYAML left as a string (e.g. "1e-4"), coercing it to float in place
        if (section, key) == ('train', 'learning_rate') and isinstance(value, str):
            try:
                value = float(value)
                config[section][key] = value
            except ValueError:
                raise TypeError(f"Error: '{section}.{key}' must be numeric")

        # bool is a subclass of int in Python; reject True/False where a number
        # is expected so a stray boolean cannot masquerade as a valid size/rate
        if expected_type != bool and isinstance(value, bool):
            type_name = "numeric" if isinstance(expected_type, tuple) else expected_type.__name__
            raise TypeError(f"Error: '{section}.{key}' must be {type_name}")

        # check if the type is correct
        if not isinstance(value, expected_type):
            type_name = "numeric" if isinstance(expected_type, tuple) else expected_type.__name__
            raise TypeError(f"Error: '{section}.{key}' must be {type_name}")

        # positivity check where required
        if must_be_positive and value <= 0:
            raise ValueError(f"Error: '{section}.{key}' must be greater than 0")

    # cross-field sanity: train_fraction is a proportion in (0, 1]
    frac = config['data']['train_fraction']
    if frac > 1:
        raise ValueError("Error: 'data.train_fraction' must be in the range (0, 1]")

    # cross-field sanity: warmup must not consume the entire training budget
    if config['train']['warmup_epochs'] >= config['train']['epochs']:
        raise ValueError("Error: 'train.warmup_epochs' must be less than 'train.epochs'")


if __name__ == "__main__":
    cfg = load_config("configs/base_config.yaml")
    try:
        validate_config(cfg)
        print(f"Success: {cfg['project_name']} now loaded")
        print(f"Backbone selected: {cfg['model']['backbone']}")
        print(f"Dataset selected: {cfg['data']['dataset']}")
    except (KeyError, ValueError, TypeError) as error:
        print(f"{error}")