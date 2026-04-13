import torch
import torch.nn as nn
from torchvision import models

class CustomEfficientNetB0(nn.Module):
    """
    blueprint for efficientnet_b0
    """
    def __init__(self, config):
        super(CustomEfficientNetB0, self).__init__()

        # Pull values from config
        embedding_dim = config['model']['embedding_dim']
        pretrained = config['model']['pretrained']
        
        # loading the backbone
        self.network = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
        
        # extracting features from the original classifier and then removing it
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Identity()
        
        # custom projection head
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # classification head
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        features = self.network(x)
        embeddings = self.embedding_layer(features)
        
        # using sigmoid for classification probabilities
        out = torch.sigmoid(self.classifier(embeddings))
        return out, embeddings

class NovelSiameseWrapper(nn.Module):
    """
    the orchestrator for the novel model
    """
    def __init__(self, backbone):
        super(NovelSiameseWrapper, self).__init__()
        self.backbone = backbone

    def forward(self, quadruplet):

        # unpacking the stack: Fake_Raw, Fake_Comp, Real_Raw, Real_Comp
        imgs = [quadruplet[:, i] for i in range(4)]
        
        # passing all four images through the same backbone object
        outputs = [self.backbone(img) for img in imgs]
        
        # separating the results (classification) and embeddings (contrastive learning)
        results = torch.stack([o[0] for o in outputs], dim=1) # [batch, 4, 1]
        embeddings = torch.stack([o[1] for o in outputs], dim=1) # [batch, 4, 128]
        
        return results, embeddings