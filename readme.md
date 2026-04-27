### Order of Execution
* add details in `base_config.yaml`
* (optional) - test if the custom written compression and filter functions work as intended - first take a fake image from the dataset and copy it to `utils/archive/` and then rename the file to `test_input.png`, then run `helpers_old.py` and finally `verify_img_processing.py`
* run `create_datasets.py` to create manipulated counterparts of raw images
* run `create_manifests.py` to create manifests
* run `train.py`
* update weights in `test.py` and run it!