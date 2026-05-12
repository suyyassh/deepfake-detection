import torch.nn as nn
import timm

class UniversalBackbone(nn.Module):
    def __init__(self, cfg):
        super(UniversalBackbone, self).__init__()
        backbone_name = cfg['model']['backbone']
        
        # 1. Load the pre-trained model as a pure feature extractor
        self.model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        # 2. Find out the feature dimension dynamically AND SAVE IT
        # CHANGE HERE: Added self.
        self.in_features = self.model.num_features 
        
        # 3. Create your custom classification head
        # CHANGE HERE: Use self.in_features
        self.fc = nn.Linear(self.in_features, 1) 

    def forward(self, x):
        features = self.model(x)  # Extract the raw embeddings
        logits = self.fc(features) # Get the fake/real prediction
        return logits, features

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