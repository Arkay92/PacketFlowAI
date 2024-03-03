import torch
import torch.nn as nn
import time  
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from scapy.all import sniff, IP, TCP, UDP, send
import re
from sklearn.model_selection import train_test_split
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
import os
import threading
from queue import Queue, Empty
from threading import Lock, Event

# Constants for feature extraction
UNCOMMON_PORT = 9999
DEFAULT_IP_VERSION = 0
DEFAULT_IP_LEN = 0
DEFAULT_TCP_SPORT = 0
DEFAULT_TCP_DPORT = UNCOMMON_PORT
DEFAULT_TCP_FLAGS = 0

FEATURE_MEAN = torch.tensor([50, 0.5])  # Example mean values for each feature
FEATURE_STD = torch.tensor([20, 0.5])   # Example std dev values for each feature
MIN_VALUE = torch.tensor([0, 0])        # Minimum value of each feature in the training set
MAX_VALUE = torch.tensor([100, 1])      # Maximum value of each feature in the training set

# Initialize locks for thread-safe operations
banned_ips_lock = Lock()
malicious_ip_counts_lock = Lock()

# Event for graceful shutdown
shutdown_event = Event()
ban_threshold = 5
attack_types = ['benign', 'DDoS', 'port_scan', 'malware', 'phishing', 'other']  # Example attack types

banned_ips = set()
no_feedback_packets = set()
malicious_ip_counts = {}

# Model file path
MODEL_FILE_PATH = 'packet_cnn_model.pth'

class PacketCNN(nn.Module):
    def __init__(self):
        super(PacketCNN, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 6)  # Assuming 6 categories including 'benign'
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
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

class TextClassifier(nn.Module):
    def __init__(self, num_categories):
        super(TextClassifier, self).__init__()
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.num_categories = num_categories

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def train_text_classifier(explanations, labels):
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(explanations, labels)
    return model

def extract_attack_type(explanation):
    keywords = {
        'DDoS': ['DDoS', 'denial of service'],
        'port_scan': ['port scan', 'port scanning'],
        'malware': ['malware', 'virus', 'trojan'],
        'phishing': ['phishing', 'spear phishing'],
        'other': ['vulnerability', 'exploit', 'unauthorized access']
    }
    
    for attack_type, keys in keywords.items():
        if any(key in explanation.lower() for key in keys):
            return attack_type
    return 'benign'

def preprocess_text_data(dataset):
    explanations = []
    attack_types = []
    for item in dataset:
        explanations.append(item['Explanation'])
        attack_type = extract_attack_type(item['Explanation'])
        attack_types.append(attack_type)
    return explanations, attack_types

def port_to_feature(port):
    port_map = {
        'ftp': 21,
        'ssh': 22,
        'http': 80,
        'https': 443
    }
    return port_map.get(port, UNCOMMON_PORT)

def flags_to_feature(flags):
    flags_map = {
        'F': 1,
        'S': 2,
        'R': 3,
        'P': 4,
        'A': 5,
        'U': 6,
        'E': 7,
        'C': 8
    }
    return sum(flags_map.get(flag, 0) for flag in flags)

def extract_features(description):
    # Extract features using regex, with error handling
    ip_version_match = re.search(r'IP version: (\d+\.\d+)', description)
    ip_version = float(ip_version_match.group(1)) if ip_version_match else DEFAULT_IP_VERSION

    ip_len_match = re.search(r'IP len: (\d+\.\d+)', description)
    ip_len = float(ip_len_match.group(1)) if ip_len_match else DEFAULT_IP_LEN

    tcp_sport_match = re.search(r'TCP sport: (\d+)', description)
    tcp_sport = float(tcp_sport_match.group(1)) if tcp_sport_match else DEFAULT_TCP_SPORT

    tcp_dport_match = re.search(r'TCP dport: (\w+)', description)
    tcp_dport = port_to_feature(tcp_dport_match.group(1)) if tcp_dport_match else DEFAULT_TCP_DPORT

    tcp_flags_match = re.search(r'TCP flags: (\w+)', description)
    tcp_flags = flags_to_feature(tcp_flags_match.group(1)) if tcp_flags_match else DEFAULT_TCP_FLAGS

    features = [ip_version, ip_len, tcp_sport, tcp_dport, tcp_flags]
    return torch.tensor(features, dtype=torch.float32)

def extract_label(explanation):
    if 'attack' in explanation or 'vulnerable' in explanation:
        return 1
    else:
        return 0

def preprocess_data(dataset):
    data = []
    targets = []
    for item in dataset:
        features = extract_features(item['Packet/Tags']).unsqueeze(0)
        label = extract_label(item['Explanation'])
        data.append(features)
        targets.append(label)
    
    data = torch.cat(data, dim=0)
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets

def packet_capture(queue, interface='eth0'):
    """Capture packets and place them into a thread-safe queue."""
    def capture(packet):
        if shutdown_event.is_set():
            return False  # Stop sniffing if shutdown is triggered
        queue.put(packet)
    sniff(iface=interface, prn=capture, stop_filter=lambda x: shutdown_event.is_set())

def process_packets(queue, model, device, optimizer, feedback_data, filter_ipv6=True, show_https=True, protocol_range=(80, 443)):
    while not shutdown_event.is_set():
        try:
            packet = queue.get(timeout=1)  # Timeout to check for shutdown event
            process_and_redirect(packet, model, device, optimizer, None, feedback_data, filter_ipv6, show_https, protocol_range)
        except Empty:  # Correctly catch the Empty exception when the queue is empty
            continue
        except Exception as e:
            print(f"Error processing packet: {e}")

def shutdown_handler():
    print("Shutdown signal received. Shutting down gracefully.")
    shutdown_event.set()

def preprocess_packet(packet):
    if not packet.haslayer(TCP) and not packet.haslayer(UDP):
        return None

    ip_version = packet.version if packet.haslayer(IP) else DEFAULT_IP_VERSION
    ip_len = packet.len if packet.haslayer(IP) else DEFAULT_IP_LEN
    tcp_sport = packet[TCP].sport if packet.haslayer(TCP) else DEFAULT_TCP_SPORT
    tcp_dport = packet[TCP].dport if packet.haslayer(TCP) else UNCOMMON_PORT

    tcp_flags = 0
    if packet.haslayer(TCP):
        tcp_flags = sum([packet[TCP].flags.F, packet[TCP].flags.S << 1, packet[TCP].flags.R << 2, packet[TCP].flags.P << 3, packet[TCP].flags.A << 4, packet[TCP].flags.U << 5, packet[TCP].flags.E << 6, packet[TCP].flags.C << 7])

    src_ip = packet[IP].src if packet.haslayer(IP) else None

    if src_ip:
        with malicious_ip_counts_lock:  # Lock before accessing the shared resource
            count = malicious_ip_counts.get(src_ip, 0) + 1  # Access the dictionary, not the lock
            malicious_ip_counts[src_ip] = count

            if count >= ban_threshold:
                with banned_ips_lock:  # Lock before modifying the banned_ips set
                    banned_ips.add(src_ip)
                    print(f"IP {src_ip} has been banned.")

    features = torch.tensor([ip_version, ip_len, tcp_sport, tcp_dport, tcp_flags], dtype=torch.float32).unsqueeze(0)
    return features

def load_feedback_file(feedback_file_path):
    feedback_data = {}
    try:
        with open(feedback_file_path, 'r') as file:
            for line in file:
                packet_id, label = line.strip().split(',')
                feedback_data[packet_id] = int(label)
    except FileNotFoundError:
        print("Warning: Feedback file not found. No feedback will be used.")
    return feedback_data

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.0f}%)\n')

def train_and_evaluate(model, device, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)
    # Save the model after training
    torch.save(model.state_dict(), MODEL_FILE_PATH)  # Save the model weights

def redirect_packet(packet):
    analysis_server_ip = '192.168.1.101'  # Ensure this IP is reachable and configured to receive packets
    if packet.haslayer(IP):
        packet[IP].dst = analysis_server_ip
        send(packet)

def update_model(model, new_data, new_labels, optimizer, device):
    model.train()
    new_data, new_labels = new_data.to(device), new_labels.to(device)
    optimizer.zero_grad()
    output = model(new_data)
    loss = nn.CrossEntropyLoss()(output, new_labels)
    loss.backward()
    optimizer.step()

def process_and_redirect(packet, model, device, optimizer, train_loader, feedback_data, filter_ipv6=True, show_https=True, protocol_range=(80, 443)):
    def strip_port(packet, src_port, dst_port):
        packet.dport = dst_port
        packet.sport = src_port

    if packet.haslayer(IP) and filter_ipv6 and packet[IP].version == 6:
        print("Skipping IPV6 packet.")
        return

    features = preprocess_packet(packet)
    if features is None:
        return  # Exit early for non-TCP/UDP packets

    features = features.to(device)
    model.eval()
    with torch.no_grad():
        output = model(features)
        prediction = torch.argmax(output, dim=1).item()

    if prediction == 0:  # Assuming 'benign' is indexed at 0
        return  # No print or action for benign packets

    if show_https and packet.haslayer(TCP) and protocol_range[0] <= packet[TCP].dport <= protocol_range[1]:
        strip_port(packet[TCP], packet[TCP].dport, 80)
    elif show_https:
        print("Skipping non-HTTPS traffic.")
        return

    if prediction != 0:
        packet_id = hashlib.sha256(bytes(packet)).hexdigest()
        attack_type = attack_types[prediction]
        print(f"Redirecting packet identified as {attack_type} attack, with ID: {packet_id}")

        if packet.haslayer(IP):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            print(f"Source IP: {src_ip}, Destination IP: {dst_ip}")
            malicious_ip_counts[src_ip] = malicious_ip_counts.get(src_ip, 0) + 1
            if malicious_ip_counts[src_ip] >= ban_threshold:
                print(f"IP {src_ip} has exceeded the ban threshold and will be banned.")
                banned_ips.add(src_ip)

        if packet.haslayer(TCP):
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
            print(f"Source Port: {src_port}, Destination Port: {dst_port}, Flags: {packet[TCP].flags}")

        if packet.haslayer(UDP):
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
            print(f"Source Port: {src_port}, Destination Port: {dst_port}")

        packet_length = len(packet)
        print(f"Packet Length: {packet_length}")

        redirect_packet(packet)

        feedback = feedback_data.get(packet_id)
        if feedback is not None:
            print(f"Using feedback for packet ID: {packet_id}. Retraining model...")
            new_data = features.unsqueeze(0)
            new_labels = torch.tensor([feedback], dtype=torch.long).to(device)
            update_model(model, new_data, new_labels, optimizer, device)
            test_loss, accuracy = evaluate(model, device, train_loader)
            print(f"After retraining - Loss: {test_loss}, Accuracy: {accuracy}%")
        elif packet_id not in no_feedback_packets:
            print(f"Collecting feedback for packet ID: {packet_id}")
            no_feedback_packets.add(packet_id)

def capture_live_packets(interface, model, device, optimizer, filter_ipv6=True, show_https=True, protocol_range=(80, 443)):
    feedback_data = load_feedback_file('packet_feedback.txt')
    print("Starting packet capture. Press Ctrl+C to stop.")
    try:
        sniff(iface=interface, prn=lambda packet: process_and_redirect(packet, model, device, optimizer, None, feedback_data, filter_ipv6, show_https, protocol_range))
    except KeyboardInterrupt:
        print("\nStopped packet capture.")

def load_and_preprocess_dataset():
    # Load the dataset from Hugging Face
    dataset = load_dataset('rdpahalavan/packet-tag-explanation')

    # Assuming the dataset has features in 'Packet/Tags' and labels in 'Explanation'
    features = []
    labels = []
    for item in dataset['train']:
        feature = extract_features(item['Packet/Tags'])
        label = extract_label(item['Explanation'])
        features.append(feature)
        labels.append(label)

    return torch.stack(features), torch.tensor(labels, dtype=torch.long)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Packet Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'capture'], required=True, help='Operation mode: train or capture')
    parser.add_argument('--interface', type=str, required=False, default='eth0', help='Network interface to capture packets from')
    parser.add_argument('--filter-ipv6', action='store_true', help='Filter IPV6 packets (default: True)')
    parser.add_argument('--show-https', action='store_true', help='Show only HTTPS related traffic (default: True)')
    parser.add_argument('--protocol', type=str, default='80:443', help='Protocol range to show (default: 80:443)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PacketCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.mode == 'train':
        # Load and preprocess the dataset
        features, labels = load_and_preprocess_dataset()

        # Split the dataset into training and testing sets
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Create instances of your custom dataset
        train_dataset = PacketDataset(features_train, labels_train)
        test_dataset = PacketDataset(features_test, labels_test)

        # Create DataLoader instances for training and testing
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Call train_and_evaluate with all required arguments
        train_and_evaluate(model, device, train_loader, test_loader)
    elif args.mode == 'capture':
        if not os.path.exists(MODEL_FILE_PATH):
            print("Model file not found. Please train the model first.")
            exit()

        model.load_state_dict(torch.load(MODEL_FILE_PATH))
        model.eval()  # Set the model to evaluation mode

        packet_queue = Queue()
        feedback_data = load_feedback_file('packet_feedback.txt')
        protocol_range = tuple(map(int, args.protocol.split(':')))  # Parse protocol_range

        # Start packet capture thread
        capture_thread = threading.Thread(target=packet_capture, args=(packet_queue, args.interface))
        capture_thread.start()

        # Start packet processing thread
        processing_thread = threading.Thread(target=process_packets, args=(packet_queue, model, device, optimizer, feedback_data, args.filter_ipv6, args.show_https, protocol_range))
        processing_thread.start()

        # Keep the main thread running until a keyboard interrupt is received
        try:
            while True:
                time.sleep(1)  # Sleep and let other threads do the work
        except KeyboardInterrupt:
            print("Shutdown signal received. Shutting down gracefully.")
            shutdown_event.set()  # Signal threads to shut down

        # Wait for threads to complete
        capture_thread.join()
        processing_thread.join()

        print("All threads have been shut down.")