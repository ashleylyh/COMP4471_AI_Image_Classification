
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_images(train_features, train_labels, num_rows=5, num_cols=5):
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(5, 5))
    
    for i in range(num_rows):
        for j in range(num_cols):
            img_index = i * num_cols + j
            if img_index >= len(train_features):  # Check to avoid index out of range
                break
            img = train_features[img_index].squeeze()
            label = train_labels[img_index]
            ax[i, j].imshow(img.permute(1, 2, 0))  # Change dimensions for display
            
            # Set title based on label
            ax[i, j].set_title("REAL" if label == 1 else "FAKE", fontsize=8)
            ax[i, j].axis('off')  # Hide axis
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2)  # Adjust wspace and hspace for spacing
    plt.tight_layout()
    plt.show()

def plot_training_time(file_path):
    metrics = pd.read_csv(file_path)
    # Bar width and positions
    bar_width = 0.35
    x = np.arange(len(metrics['Model']))

    # Create the bars
    plt.figure(figsize=(8, 4))
    plt.bar(x - bar_width/2, metrics['Adam_Time'], width=bar_width, label='Adam', color='#195190')
    plt.bar(x + bar_width/2, metrics['SGD_Time'], width=bar_width, label='SGD', color='#9E1030')

    # Labels and title
    plt.xlabel('Model')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time for Different Models by Optimizer')
    plt.xticks(x, metrics['Model'])
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_result(data, model_names, optimizer_names, model_colors, graph_title, y_label):

    plt.figure(figsize=(9, 6))
    
    for model in model_names:
        for optimizer in optimizer_names:
            key = f"{model}_{optimizer}"
            if key in data:
                line_style = '-' if optimizer == 'Adam' else 'dashed'  # Line for Adam, dot for SGD
                plt.plot(data[key], label=f"{model} - {optimizer}", color=model_colors[model], linestyle=line_style)

    plt.xlabel('Epoch Number')
    plt.ylabel(y_label)
    plt.title(graph_title)
    plt.xticks(range(len(data[key])), range(1, len(data[key]) + 1))  # Epochs as ticks
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(ax, csv_file, labels, model_name):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Extract true and predicted labels
    true_labels = data['y_true'].values
    pred_labels = data['y_pred'].values

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plotting the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_ylabel('True label', fontsize=10)
    ax.set_xlabel('Predicted label', fontsize=10)
    ax.set_title(f'{model_name}', fontsize=12)


def plot_model_metrics(metrics_df):
    # Plotting
    plt.figure(figsize=(8, 4))  # Increase figure size

    # Define colors for each metric
    colors = {
        'Accuracy': '#ed5314',  # Steel Blue
        'F1 Score': '#ffb92a',   # Slate Blue
        'Precision': '#9bca3e',   # Medium Sea Green
        'Recall': '#3abbc9'       # Rosy Brown
    }

    # Plot each metric
    for metric in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
        plt.plot(metrics_df['Model'], metrics_df[metric], marker='o', label=metric, color=colors[metric])
        for i, v in enumerate(metrics_df[metric]):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', color=colors[metric])

    # Customize plot
    plt.title('Model Performance: Accuracy, F1 Score, Precision, and Recall')
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.ylim(0.5, 1)  # Focus on the upper part of the scale
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
