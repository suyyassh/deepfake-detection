import torch
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
        
        # pull the dimension from your config
        proj_dim = config['model']['embedding_dim']
        
        # custom projection head for the contrastive Push/Pull loss
        self.projection_head = nn.Sequential(
            nn.Linear(self.backbone.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

    def forward(self, quadruplet):
        # unpacking the stack: fake_raw, fake_comp, real_raw, real_comp
        imgs = [quadruplet[:, i] for i in range(4)]
        
        outputs = [self.backbone(img) for img in imgs]
        
        # stack the BCE classifications [batch, 4, 1]
        results = torch.stack([o[0] for o in outputs], dim=1) 
        
        # pass the 1280D features through the projection head
        embeddings = torch.stack([self.projection_head(o[1]) for o in outputs], dim=1) 
        
        return results, embeddings