import torch.nn as nn
from torchvision import models

class CustomEfficientNetB0(nn.Module):
    def __init__(self, config):
        super(CustomEfficientNetB0, self).__init__()
        pretrained = config['model']['pretrained']
        
        # loading the backbone
        self.network = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        
        # extract the 1280 feature dimension and remove the default classifier
        self.in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Identity()
        
        # classification head now sits directly on the robust 1280D features
        self.classifier = nn.Linear(self.in_features, 1)

    def forward(self, x):
        features = self.network(x)
        out = self.classifier(features)
        return out, features

class NovelSiameseWrapper(nn.Module):
    def __init__(self, backbone, config):
        super(NovelSiameseWrapper, self).__init__()
        self.backbone = backbone
        
        proj_dim = config['model']['embedding_dim']
        
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, quadruplet):
        # quadruplet shape: [batch_size, 4, 3, 256, 256]
        batch_size = quadruplet.size(0)
        
        # CRITICAL FIX: Fold the 4 images into the batch dimension
        # Shape becomes: [batch_size * 4, 3, 256, 256] -> e.g., 64 images
        flat_imgs = quadruplet.reshape(-1, 3, quadruplet.size(3), quadruplet.size(4))
        
        # Pass all 64 images through the backbone simultaneously (Stable BatchNorm!)
        flat_results, flat_features = self.backbone(flat_imgs)
        
        # Pass features through projection head
        flat_embeddings = self.projection_head(flat_features)
        
        # Unfold the results back into the Siamese shape: [batch_size, 4, ...]
        results = flat_results.view(batch_size, 4, 1)
        embeddings = flat_embeddings.view(batch_size, 4, -1)
        
        return results, embeddings