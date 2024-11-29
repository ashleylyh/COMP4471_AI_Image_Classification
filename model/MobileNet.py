import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetTransferLearning(nn.Module):
    def __init__(self, img_height, img_width):
        super(MobileNetTransferLearning, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Load the MobileNetV3 model pre-trained on ImageNet
        self.mobilenet_base = models.mobilenet_v3_small(weights='IMAGENET1K_V1')  # Load the MobileNetV3 model
        self.mobilenet_base.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, 576),
        )
        self.mobilenet_base.fc = nn.Identity() 
        self.mobilenet_base.trainable = True  # Fine-tune MobileNetV3

        # Define new layers
        self.batch_norm = nn.BatchNorm1d(576)  # Update based on MobileNetV3 output
        self.fc1 = nn.Linear(576, 256)  # Adjust input size for the first fully connected layer
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.mobilenet_base(x)
        # x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = self.batch_norm(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)

