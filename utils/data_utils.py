
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# counting the number of each class in the dataset
def count_classes(dataloader):
    real, fake = 0, 0
    for images, labels in tqdm(dataloader):
        for label in labels:
            if label == 1:
                real += 1
            else:
                fake += 1
    print(f"REAL: {real}, FAKE: {fake}")

# write a funciton to save train_losses, val_accuracies, and train_acc into a 2 csv file
def save_metrics(model_name, train_losses, val_accuracies, train_acc):
    metrics1 = pd.DataFrame({
        'train_losses': train_losses,
    })
    metrics1.to_csv(f'./training_result/{model_name}_TrainLoss.csv', index=False)
    metrics2 = pd.DataFrame({
        'train_acc': train_acc,
    })
    metrics2.to_csv(f'./training_result/{model_name}_TrainAcc.csv', index=False)
    metrics3 = pd.DataFrame({
        'val_accuracies': val_accuracies,
    })
    metrics3.to_csv(f'./training_result/{model_name}_ValAcc.csv', index=False)


def save_training_times_to_csv(filename):
    # Data preparation
    data = {
        'Model': ['VGG', 'ResNet',  'MobileNet', 'EfficientNet','ViT',  'ELA+Mobile', 'ELA+Eff'],
        'Adam_Time': [2509.4, 1676.2, 640.5, 1837.0, 3511.4, 727.6, 2126.3],
        'SGD_Time': [2533.0, 1645.6, 644.8, 1814.6, 4710.10, 691.6, 1900.7]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)


def load_train_data(file_path):
    """
    Load training loss data from a CSV file.
    
    Parameters:
    - file_path: Path to the CSV file.
    
    Returns:
    - A pandas Series with loss values.
    """
    epoch_loss = pd.read_csv(file_path).squeeze()  # Assuming no header in CSV
    epoch_loss = epoch_loss.values.reshape(-1, 625).mean(axis=1)  # Average loss per epoch
    return epoch_loss

def load_val_data(file_path):
    """
    Load validation accuracy data from a CSV file.
    
    Parameters:
    - file_path: Path to the CSV file.
    
    Returns:
    - A pandas Series with validation accuracy values.
    """
    val_acc = pd.read_csv(file_path).squeeze()  # Assuming no header in CSV
    return val_acc

# write a function to have the result into csv file
def save_result_to_csv(y_true, y_pred, model_name, file_name):
    result = {'y_true': y_true, 'y_pred': y_pred}
    df = pd.DataFrame(result)
    df.to_csv(file_name, index=False)
    print(f"Result for {model_name} saved to {file_name}")

# write a function to calculate accuracy, precision, recall, and f1 score
def calculate_metrics(file_path, model_name):
    df = pd.read_csv(file_path)
    y_true = df['y_true']
    y_pred = df['y_pred']
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # print(f"{model_name}: Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    return accuracy, precision, recall, f1_score



def save_tuning_results(results, filename):
    """
    Save hyperparameter tuning results to a CSV file.

    Parameters:
    results (list of dict): The tuning results from hyperparameter tuning.
    filename (str): The name of the file to save the results to.
    """
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    # exclude the loss
    df = df.drop(columns=['train_losses'])
    # Add a column for the best validation accuracy
    df['best_val_accuracy'] = df['val_accuracies'].apply(lambda x: max(x))
    # Add a column for the best training accuracy
    df['best_train_accuracy'] = df['train_accuracies'].apply(lambda x: max(x))
    # Add a column for the average training accuracy
    df['average_train_accuracy'] = df['train_accuracies'].apply(lambda x: np.mean(x))
    # Add a column for the average validation accuracy
    df['average_val_accuracy'] = df['val_accuracies'].apply(lambda x: np.mean(x))
    # drop the train and validation accuracy
    df = df.drop(columns=['train_accuracies', 'val_accuracies'])


    # Save DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"Tuning results saved to {filename}")