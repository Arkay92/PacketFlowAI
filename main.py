import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_dataset
from scapy.all import sniff
from scapy.layers.inet import TCP, UDP

FEATURE_MEAN = torch.tensor([50, 0.5])  # Example mean values for each feature
FEATURE_STD = torch.tensor([20, 0.5])   # Example std dev values for each feature
MIN_VALUE = torch.tensor([0, 0])        # Minimum value of each feature in the training set
MAX_VALUE = torch.tensor([100, 1])      # Maximum value of each feature in the training set

class PacketCNN(nn.Module):
    def __init__(self):
        super(PacketCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PacketDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        if self.transform:
            x = self.transform(x)
        return x, y

def preprocess_data(dataset):
    data = [item['feature'] for item in dataset]
    targets = [item['label'] for item in dataset]
    data = torch.tensor(data, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets

def normalize(features):
    return (features - FEATURE_MEAN) / FEATURE_STD

def scale(features):
    return (features - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)

def preprocess_packet(packet):
    packet_length = len(packet)
    protocol_type = 0 if packet.haslayer(TCP) else 1 if packet.haslayer(UDP) else -1
    features = torch.tensor([packet_length, protocol_type], dtype=torch.float32)
    features = normalize(features)
    features = scale(features)
    return features.view(1, 1, 1, -1)

def process_and_predict(packet, model, device):
    try:
        features = preprocess_packet(packet)
        tensor_features = features.unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(tensor_features)
            prediction = output.argmax(dim=1).item()
            print(f"Predicted class: {prediction}")
    except Exception as e:
        print(f"Error processing packet: {e}")

def train(model, device, train_loader, optimizer):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return test_loss, accuracy

def train_and_evaluate(model, device):
    raw_dataset = load_dataset('rdpahalavan/packet-tag-explanation')
    train_data, train_targets = preprocess_data(raw_dataset['train'])
    test_data, test_targets = preprocess_data(raw_dataset['test'])
    train_dataset = PacketDataset(train_data, train_targets)
    test_dataset = PacketDataset(test_data, test_targets)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')

    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer)
        test_loss, _ = evaluate(model, device, test_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'packet_cnn_model.pth')
            print("Model saved successfully!")

def capture_live_packets(interface, num_packets, model, device):
    sniff(iface=interface, prn=lambda packet: process_and_predict(packet, model, device), count=num_packets)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Packet Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'capture'], required=True, help='Operation mode: train or capture')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PacketCNN().to(device)

    if args.mode == 'train':
        train_and_evaluate(model, device)
    elif args.mode == 'capture':
        capture_live_packets('eth0', 10, model, device)
