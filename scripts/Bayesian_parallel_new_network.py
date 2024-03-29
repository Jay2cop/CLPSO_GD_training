import torch
import torchvision
import numpy as np
import torch.multiprocessing as mp
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import precision_score
import os
from bayes_opt import BayesianOptimization
import torch.nn.functional as F

os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:torchvision.io.image'

class my_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(my_CNN, self).__init__()
        
        # Convolutional Blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.5)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.batch_norm_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional Blocks
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = self.maxpool3(x)
        x = self.dropout3(x)

        # Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.batch_norm_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


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
    model = my_CNN()
    num_ftrs = model.fc2.in_features
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        
        weight_part = weights_tensor[:10*num_ftrs].view(10, num_ftrs)
        bias_part = weights_tensor[10*num_ftrs:]

        original_weight = model.fc2.weight.data.clone()
        original_bias = model.fc2.bias.data.clone()

        model.fc2.weight.data = weight_part
        model.fc2.bias.data = bias_part

        val_loss = 0.0
        for data in val_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

        model.fc2.weight.data = original_weight
        model.fc2.bias.data = original_bias

        return val_loss / len(val_loader.dataset)

def worker(particle, model_path, device, val_loader, criterion, fitnesses, index):
    fitness = evaluate(particle, model_path, device, val_loader, criterion)
    fitnesses[index] = fitness
    print(f"Particle {index + 1} evaluated. Fitness: {fitness:0.4f}")

def run_clpso(model_path, val_loader, criterion, w, c1, c2, fine_tune_epochs=2, num_particles=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = 0.1

    # Load the model to determine its structure
    model = my_CNN()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc2.parameters():
        param.requires_grad = True

    model.fc2.reset_parameters()
    model = model.to(device)
    num_output_neurons = 10
    num_ftrs = model.fc2.in_features
    total_params = (num_ftrs * num_output_neurons) + num_output_neurons

    # Initialize particles and their velocities
    particles = [np.random.uniform(-1, 1, total_params) for _ in range(num_particles)]
    velocities = [np.zeros(total_params) for _ in range(num_particles)]
    personal_best_positions = [np.copy(p) for p in particles]
    personal_best_scores = [float('inf') for _ in range(num_particles)]
    global_best_position = np.random.uniform(-1, 1, total_params)
    global_best_score = float('inf')

    # Early stopping
    early_stopping = EarlyStopping(patience=5)
    manager = mp.Manager()
    fitnesses = manager.list([0] * num_particles)
    model.train()
    for epoch in range(fine_tune_epochs):
        for i in range(num_particles):
            r1, r2 = np.random.rand(total_params), np.random.rand(total_params)
            for d in range(total_params):
                selected_particle = np.random.choice(num_particles)
                velocities[i][d] = w * velocities[i][d] + c1 * r1[d] * (personal_best_positions[selected_particle][d] - particles[i][d]) + c2 * r2[d] * (global_best_position[d] - particles[i][d])
                velocities[i][d] = np.clip(velocities[i][d], -bounds, bounds)

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], -1, 1)

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
            weight_part = global_best_tensor[:-10].view_as(model.fc2.weight)
            bias_part = global_best_tensor[-10:]
            model.fc2.weight.data.copy_(weight_part)
            model.fc2.bias.data.copy_(bias_part)

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
    return global_best_score, model

def pso_performance(w, c1, c2, model_path, val_loader, criterion):
    best_fitness, optimized_model = run_clpso(model_path, val_loader, criterion, w=w, c1=c1, c2=c2, fine_tune_epochs=100, num_particles=100)
    return -best_fitness

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = '/school/intelligence_coursework/new_CNN/trained_network/new_CNN_notebook.pth'

    # Load and preprocess CIFAR-10 data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=0)

    val_loader = testloader
    criterion = nn.CrossEntropyLoss()

    pbounds = {'w': (0.4, 0.9), 'c1': (1.0, 2.0), 'c2': (1.0, 2.0)}
    optimizer = BayesianOptimization(
        f=lambda w, c1, c2: pso_performance(w, c1, c2, model_path, val_loader, criterion),
        pbounds=pbounds,
        random_state=1,
    )

    # Perform optimization
    optimizer.maximize(init_points=2, n_iter=10)

    # Print the best result
    print(optimizer.max)

    