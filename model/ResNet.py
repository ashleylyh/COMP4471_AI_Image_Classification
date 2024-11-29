import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary



class ResNetTransferLearning(nn.Module):
    def __init__(self, img_height, img_width):
        super(ResNetTransferLearning, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Load the ResNet50 model pre-trained on ImageNet
        self.resnet_base = models.resnet50(weights='IMAGENET1K_V2')
        self.resnet_base.fc = nn.Identity() 
        self.resnet_base.trainable = True # Fine-tune ResNet50

        # Define new layers
        self.batch_norm = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.resnet_base(x)
        x = self.batch_norm(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        return x

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
    

