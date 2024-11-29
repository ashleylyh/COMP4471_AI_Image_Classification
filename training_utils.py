import torch
from tqdm import tqdm
import itertools


def check_accuracy(model, deviceUse, validationLoader):
    # print('Checking accuracy on val set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in validationLoader:
            x = x.to(device=deviceUse)  # move to device, e.g. GPU
            y = y.to(device=deviceUse)
            scores = model(x)
            # Assuming binary classification output
            preds = (scores >= 0.5).float()  # Convert scores to binary predictions

            num_correct += (preds.view(-1) == y.view(-1)).sum().item()  # Ensure both are flat
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples if num_samples > 0 else 0  # Avoid division by zero
    return 100 * acc


def train_loop(model, optimizer, trainLoader, validationLoader, deviceUse, epochs=1):
    model = model.to(device=deviceUse)  # move the model parameters to CPU/GPU
    train_losses = []
    val_accuracies = []
    train_acc = []

    for e in range(1, epochs + 1):
        model.train()  # put model to training mode
        num_correct = 0
        num_samples = 0
        total_loss = 0

        with tqdm(trainLoader, unit="batch", desc=f"Epoch {e}/{epochs}") as tepoch:
            for t, (x, y) in enumerate(tepoch):
                x = x.to(device=deviceUse)  # move to device, e.g. GPU
                y = y.to(device=deviceUse).view(-1, 1).float()  # Ensure y has correct shape

                optimizer.zero_grad()  # Zero out gradients
                scores = model(x)
                loss = torch.nn.BCELoss()(scores, y)  # Compute loss

                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                train_losses.append(loss.item())
                total_loss += loss.item()

                # Calculate training accuracy
                preds = (scores >= 0.5).float()  # Convert scores to binary predictions
                num_correct += (preds.view(-1) == y.view(-1)).sum().item()
                num_samples += preds.size(0)

                train_accuracy = num_correct / num_samples * 100
                train_acc.append(train_accuracy)

                tepoch.set_postfix(loss=loss.item(), train_accuracy=train_accuracy)

        # Check validation accuracy at the end of the epoch
        v_accuracy = check_accuracy(model, deviceUse, validationLoader)
        val_accuracies.append(v_accuracy)
        print(f"Epoch {e} validation accuracy: {v_accuracy:.2f}%")
        
        # Optionally add a learning rate scheduler here

    return train_losses, val_accuracies, train_acc


# Function to load the model weights
def load_model(model_class, model_path, deviceUse):
    model = model_class(img_height=224, img_width=224)  # Initialize the model using the provided class
    if deviceUse.type == "cpu":
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path,  weights_only=True, map_location=deviceUse))  # Load weights
    model.to(deviceUse)  # Move model to the appropriate device
    return model

def test_model(model, deviceUse, testLoader):
    model.eval()  # set model to evaluation mode
    true_labels = []
    pred_labels = []
    
    with torch.no_grad():
        for x, y in testLoader:
            x = x.to(device=deviceUse)  # move to device, e.g. GPU
            y = y.to(device=deviceUse).view(-1, 1).float()  # Ensure y has correct shape
            
            scores = model(x)
            preds = (scores >= 0.5).float()  # Convert scores to binary predictions
            
            true_labels.extend(y.view(-1).cpu().numpy())  # Store true labels
            pred_labels.extend(preds.view(-1).cpu().numpy())  # Store predicted labels

    return true_labels, pred_labels


def hyperparameter_tuning(model, trainLoader, validationLoader, deviceUse, epochs=10):
    # Define hyperparameter options
    learning_rates = [0.01, 0.001, 0.0001]
    weight_decays = [0.005, 0.0005]
    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        # Add other optimizers if needed
    }

    results = []

    # Generate all combinations of hyperparameters
    for lr, wd, optimizer_name in itertools.product(learning_rates, weight_decays, optimizers.keys()):
        # Create optimizer
        optimizer = optimizers[optimizer_name](model.parameters(), lr=lr, weight_decay=wd)

        print(f"Training with optimizer: {optimizer_name}, learning rate: {lr}, weight decay: {wd}")

        # Train the model
        train_losses, val_accuracies, train_accuracies = train_loop(model, optimizer, trainLoader, validationLoader, deviceUse, epochs)

        # Store results
        results.append({
            'optimizer': optimizer_name,
            'learning_rate': lr,
            'weight_decay': wd,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'train_accuracies': train_accuracies,
        })

    return results