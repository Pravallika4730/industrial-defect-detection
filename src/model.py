import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        # Load ImageNet-pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT

        backbone = resnet18(weights=weights)

        # Remove the final classification layer
        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # Freeze pretrained weights
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def forward(self, x):

        x = self.features(x)

        # [batch, 512, 1, 1]
        #        ↓
        # [batch, 512]

        x = torch.flatten(x, 1)

        return x


# ==========================================
# TEST MODEL
# ==========================================

if __name__ == "__main__":

    model = ResNet18FeatureExtractor()

    model.eval()

    print("=" * 60)
    print("Pretrained ResNet18 Feature Extractor")
    print("=" * 60)

    sample = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        features = model(sample)

    print(f"\nInput shape:   {sample.shape}")
    print(f"Feature shape: {features.shape}")

    print("\nPretrained model test completed successfully! ✅")