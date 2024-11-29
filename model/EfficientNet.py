import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetTransferLearning(nn.Module):
    def __init__(self, img_height, img_width):
        super(EfficientNetTransferLearning, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Load the EfficientNetV2 model pre-trained on ImageNet
        self.efficientnet_base = models.efficientnet_v2_s(weights='IMAGENET1K_V1')  # Use EfficientNetV2-S
        self.efficientnet_base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
        )
        self.efficientnet_base.fc = nn.Identity() 
        self.efficientnet_base.trainable = True  # Fine-tune EfficientNetV2

        # Define new layers
        self.batch_norm = nn.BatchNorm1d(512)  # Update based on EfficientNetV2 output
        self.fc1 = nn.Linear(512, 256)  # Adjust input size for the first fully connected layer
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.efficientnet_base(x)
        # x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = self.batch_norm(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

