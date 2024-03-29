import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from bayes_opt import BayesianOptimization

# Function to add L1 regularization
def add_l1_regularization(model, loss, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss = loss + lambda_l1 * l1_norm
    return loss

def train_evaluate(l1, l2):
    # Initialize the model, criterion, and optimizer
    print(f"Training with l1: {l1}, l2: {l2}")
    model = my_CNN(num_classes=10).to(device)

    # Freeze all layers except fc2 and reset its parameters
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc2.parameters():
        param.requires_grad = True
    model.fc2.reset_parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.fc2.parameters(), lr=0.00018, weight_decay=l2)

    early_stopping = EarlyStopping()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = add_l1_regularization(model, loss, l1)
            loss.backward()
            optimizer.step()
        print(f"Completed training epoch {epoch + 1}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
        
        # Early Stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    print(f"Finished training with l1: {l1}, l2: {l2}, Final Validation Loss: {val_loss:.4f}")
    return val_loss

# Bayesian Optimization
def optimize_hyperparameters(train_evaluate):
    def optimize(l1, l2):
        return -train_evaluate(l1, l2)  # Negative because Bayesian Optimization maximizes the function

    optimizer = BayesianOptimization(
        f=optimize,
        pbounds={'l1': (0, 0.1), 'l2': (0, 0.001)},
        random_state=1,
    )
    optimizer.maximize(init_points=2, n_iter=5)
    return optimizer.max

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

# Early Stopping Class
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

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # New addition
    transforms.RandomRotation(15),  # Increased rotation
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),  # Slightly increased
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # New addition
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=8)

classes = trainset.classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = my_CNN(num_classes=10)
model.load_state_dict(torch.load('/school/intelligence_coursework/new_CNN/trained_network/new_CNN_notebook.pth'))
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc2.parameters(), lr = 0.00018, weight_decay = 0.01)

# List to store average loss per epoch
epoch_losses = []

# Initialize early stopping
early_stopping = EarlyStopping()

#validation loss
val_loader = DataLoader(testset, batch_size=512, shuffle=True)

num_epochs = 50

best_params = optimize_hyperparameters(train_evaluate)
print(f"Best Hyperparameters: {best_params}")