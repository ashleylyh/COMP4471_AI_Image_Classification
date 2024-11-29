import os
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

class Image_Dataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        # Load images and labels from the directory structure
        for label in ['REAL', 'FAKE']:
            label_dir = os.path.join(base_dir, label)
            for img_file in os.listdir(label_dir):
                if img_file.endswith('.jpg'):  # Assuming you only want .jpg files
                    self.images.append(os.path.join(label_dir, img_file))
                    self.labels.append(1 if label == 'REAL' else 0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB to handle different image modes
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)
    
    def split_dataset(self, train_size=0.8):
            # Get the indices of the images
            indices = list(range(len(self)))
            # Get the labels of the images
            labels = [self.labels[i] for i in indices]
            # Split the indices into train and validation indices
            train_indices, val_indices = train_test_split(indices, train_size=train_size, stratify=labels)
            # Create the train and validation subsets
            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)
            return train_dataset, val_dataset