import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def get_transforms(config):
    """
    creates a standard transformation pipeline based on the config.
    ensures img_size is synced across the whole project.
    """
    size = config['data']['img_size'] 
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),

        # standard normalisation 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class StandardDataset(Dataset):
    """
    used for training the baseline model and testing both models
    """
    def __init__(self, manifest_path, config):
        self.data = pd.read_csv(manifest_path)
        self.transform = get_transforms(config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        label = row['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class QuadrupletDataset(Dataset):
    """
    used for training the novel model
    """
    def __init__(self, manifest_path, config):
        self.data = pd.read_csv(manifest_path)
        self.transform = get_transforms(config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # paths from the manifest
        paths = [row['fake_fpr'], row['fake_fpr_comp'], row['real_raw'], row['real_comp']]
        imgs = [Image.open(p).convert('RGB') for p in paths]

        if self.transform:
            imgs = [self.transform(img) for img in imgs]

        # stack into [4, 3, size, size]
        img_stack = torch.stack(imgs)
        
        # appying labels: fake, fake, real, real
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32)

        return img_stack, labels