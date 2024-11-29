import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, img_height, img_width):
        super(CNNModel, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        
        # Define the CNN architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1)  # 1st Conv Layer
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # 1st Max Pooling Layer
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1)  # 2nd Conv Layer
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # 2nd Max Pooling Layer
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1)  # 3rd Conv Layer
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))  # 3rd Max Pooling Layer

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1)  # 3rd Conv Layer
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))  # 3rd Max Pooling Layer
        
        self.flatten = nn.Flatten()  # Flatten Layer

        # self.fc1 = nn.Linear(16 * (img_height // 8) * (img_width // 8), 256)  # Fully Connected Layer 1
        self.fc1 = nn.Linear(2304, 256)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 64)  # Output layer
        self.output = nn.Linear(64, 1)


    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = self.pool1(x)
        
        x = nn.ReLU()(self.conv2(x))
        x = self.pool2(x)
        
        x = nn.ReLU()(self.conv3(x))
        x = self.pool3(x)

        x = nn.ReLU()(self.conv4(x))
        x = self.pool4(x)
        
        x = self.flatten(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.ReLU()(self.fc2(x))
        x = torch.sigmoid(self.output(x))
        
        return x

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
