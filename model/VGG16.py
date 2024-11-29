import torch
import torch.nn as nn
import torchvision.models as models

class VGGTransferLearning(nn.Module):
    def __init__(self, img_height, img_width):
        super(VGGTransferLearning, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Load the VGG16 model pre-trained on ImageNet
        self.vgg_base = models.vgg16(weights='IMAGENET1K_V1')
        self.vgg_base.classifier = nn.Sequential(
            nn.Linear(25088, 4096),  # Adjust based on the input size of the previous layer
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.vgg_base.fc = nn.Identity() 
        self.vgg_base.trainable = True  # Fine-tune VGG16

        # Define new layers
        self.batch_norm = nn.BatchNorm1d(512)  # BatchNorm after VGG output
        self.fc1 = nn.Linear(512, 256)  # Adjust for VGG output size
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.vgg_base(x)
        x = self.batch_norm(x)
        # x = torch.flatten(x, 1)  # Flatten the output for the fully connected layers
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
