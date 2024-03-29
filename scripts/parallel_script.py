import torch
import torchvision
import numpy as np
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import copy
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:torchvision.io.image'

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def evaluate(weights, model_path, device, val_loader, criterion):
    # Load the model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 10)  # Assuming 10 output classes
    )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    with torch.no_grad():
        # Ensure weights_tensor has the correct size
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        
        # Reshape weights
        weight_part = weights_tensor[:10*num_ftrs].view(10, num_ftrs)  # Reshaping to [10, 512]
        bias_part = weights_tensor[10*num_ftrs:]

        original_weight = model.fc[1].weight.data.clone()
        original_bias = model.fc[1].bias.data.clone()

        model.fc[1].weight.data = weight_part
        model.fc[1].bias.data = bias_part

        val_loss = 0.0
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

        model.fc[1].weight.data = original_weight
        model.fc[1].bias.data = original_bias

        return val_loss / len(val_loader.dataset)

def worker(particle, model_path, device, val_loader, criterion, fitnesses, index):
    fitness = evaluate(particle, model_path, device, val_loader, criterion)
    fitnesses[index] = fitness
    print(f"Particle {index + 1} evaluated. Fitness: {fitness:0.4f}")

# Assigning classes
def indices_to_class_names(indices, class_names):
    return [class_names[i] for i in indices]

def run_clpso(model_path, val_loader, criterion, fine_tune_epochs=200, num_particles=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w=0.8284808327884073
    #w_begin = 0.9
    #w_finish = 0.4
    c1, c2 = 1.3224727594368535, 1.7635354487773103
    bounds = 0.1

    # Load the model to determine its structure
    model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, 10))
    model.fc[1].reset_parameters()  # Randomize weights of fully connected layer
    model = model.to(device)
    num_output_neurons = 10  # Assuming 10 output classes
    total_params = (num_ftrs * num_output_neurons) + num_output_neurons  # Total parameters in the FC layer

    # Initialize particles and their velocities
    particles = [np.random.uniform(-1, 1, total_params) for _ in range(num_particles)]
    velocities = [np.zeros(total_params) for _ in range(num_particles)]
    personal_best_positions = [np.copy(p) for p in particles]
    personal_best_scores = [float('inf') for _ in range(num_particles)]
    global_best_position = np.random.uniform(-1, 1, total_params)
    global_best_score = float('inf')

    # Early stopping
    early_stopping = EarlyStopping(patience=8)
    manager = mp.Manager()
    fitnesses = manager.list([0] * num_particles)

    for epoch in range(fine_tune_epochs):
        #w = w_begin - (epoch / fine_tune_epochs) * (w_begin - w_finish)
        for i in range(num_particles):
            r1, r2 = np.random.rand(total_params), np.random.rand(total_params)
            for d in range(total_params):
                selected_particle = np.random.choice(num_particles)  # Select a random particle for each dimension
                velocities[i][d] = w * velocities[i][d] + c1 * r1[d] * (personal_best_positions[selected_particle][d] - particles[i][d]) + c2 * r2[d] * (global_best_position[d] - particles[i][d])
                velocities[i][d] = np.clip(velocities[i][d], -bounds, bounds)  # Apply velocity clipping

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], -1, 1)  # Apply bounds to particle position

        processes = []
        for i, particle in enumerate(particles):
            p = mp.Process(target=worker, args=(particle, model_path, device, val_loader, criterion, fitnesses, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Update personal and global bests
        for i, fitness in enumerate(fitnesses):
            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = particles[i].copy()
            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = particles[i].copy()
        with torch.no_grad():
            global_best_tensor = torch.from_numpy(global_best_position).float().to(device)
            weight_part = global_best_tensor[:-10].view_as(model.fc[1].weight)
            bias_part = global_best_tensor[-10:]
            model.fc[1].weight.data.copy_(weight_part)
            model.fc[1].bias.data.copy_(bias_part)

        # Validation and Early Stopping
        model.eval()
        val_loss = 0.0
        all_predict = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predict = torch.max(outputs, 1)
                all_predict.extend(predict.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        precision = precision_score(all_labels, all_predict, average='weighted', zero_division=1)

        print(f"Epoch {epoch + 1}/{fine_tune_epochs} - Validation Loss: {val_loss:0.4f}, Precision: {precision:0.4f}")
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        print(f"Epoch {epoch+1}/{fine_tune_epochs} - Best Global Fitness: {global_best_score:0.4f}")

    print(f"Optimization completed. Best Global Fitness: {global_best_score:0.4f}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
     # Set CUDA and multiprocessing start method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define and load model
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout layer
        nn.Linear(num_ftrs, 10)  # Fully connected layer
    )
    model.load_state_dict(torch.load('/school/intelligence_coursework/Trained_networks/model_state_dict_2.pth', map_location=torch.device('cpu')))
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)

    # Load and preprocess CIFAR-10 data
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    
    run_clpso('/school/intelligence_coursework/Trained_networks/model_state_dict_2.pth', testloader, criterion, fine_tune_epochs=150, num_particles=30)

    # Ensure the model is in evaluation mode
    model.eval()

    # Initialize metrics
    correct = 0
    total = 0
    classes = trainset.classes
    class_correct = {classname: 0 for classname in classes}
    class_total = {classname: 0 for classname in classes}

    # Accuracy on the whole test set
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[classes[label.item()]] += 1
                class_total[classes[label.item()]] += 1

    print('Accuracy of the network on the test images: %f %%' % (100 * correct / total))

    # Accuracy for each class
    for classname, correct_count in class_correct.items():
        accuracy = 100 * float(correct_count) / class_total[classname]
        print(f'Accuracy for class {classname:5s} is {accuracy:0.1f} %')

    # Compute Precision, Recall, F1 Score
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Apply the mapping to all labels and predictions
    all_labels_names = indices_to_class_names(all_labels, classes)
    all_predictions_names = indices_to_class_names(all_predictions, classes)

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

