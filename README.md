# HKUST COMP 4471 & ELEC 4240 Course Project: 

# Distinguishing Reality: Classifying Real vs. AI-Generated Images


This repository contains the implementation of our project classifying Real vs AI-Generated image. The project is part of the course COMP 4471 & ELEC 4240: Deep Learning in Computer VIsion at the Hong Kong University of Science and Technology (HKUST). 

We aim to classify real images from AI-generated images using some pretrained CNN, vision transformer. We also explore the effectiveness of ELA images in the classification task.

For more details on the project objectives, methodology, results, and analysis, please refer to the **[Project Report](./....pdf)**.

## ⚙️ Environment Setup
1. Run the following command to create a new conda environment with the required dependencies:
```conda create -n <your_env_name> python=3.10```
2. Activate the conda environment:
```conda activate <your_env_name>```
3. Install the required dependencies:
```pip install -r requirements.txt```
4. Deactivate the conda environment:
```conda deactivate```

## 📂 Repository Structure
- **`models/`**: Contains the models structure implemented by PyTorch.
- **`utils/`**: Contains the utility functions for data preprocessing, model training, and visualization.
- **`ela_image_conversion.ipynb`**: Contains the code for converting images to ELA images.
- **`model_compare.ipynb`**: Contains the code for training and evaluating the image classification model.
  
## 📦 Dataset
- Due to the large size of the dataset, the images are not included in the repository. Please download the dataset from the following link: [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- The ELA images are generated using the `ela_image_conversion.ipynb` notebook.

## 📦 Results and Model Weights
- Due to the large size of the model weights, the weights are not included in the repository. 
- Due to the number of result files, the results are not included in the repository.

## 👨‍💻 Authors
- **LI, Yu-hsi** (SID: 20819823, ITSC: ylils)
- **HUANG, I-wei** (SID: 20824074, ITSC: ihuangaa)
- **KUO, Chen-chieh** (SID: 20825315, ITSC: ckuoab)

