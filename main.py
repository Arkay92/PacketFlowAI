import logging, time, torch, re, hashlib, os, threading, signal
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from scapy.all import sniff, IP, TCP, UDP, send
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from queue import Queue, Empty
from threading import Lock, Event
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import numpy as np

# Constants for feature extraction
UNCOMMON_PORT = 9999
DEFAULT_IP_VERSION = 0
DEFAULT_IP_LEN = 0
DEFAULT_TCP_SPORT = 0
DEFAULT_TCP_DPORT = UNCOMMON_PORT
DEFAULT_TCP_FLAGS = 0

# Hypervector dimensions
HV_DIMENSION = 10000  # Dimension of the hypervectors
NUM_LEVELS = 100  # Number of levels for numerical features

# Initialize locks for thread-safe operations
banned_ips_lock = Lock()
malicious_ip_counts_lock = Lock()

# Event for graceful shutdown
shutdown_event = Event()
attack_types = ['benign', 'DDoS', 'port_scan', 'malware', 'phishing', 'other']  # Example attack types

banned_ips = set()
no_feedback_packets = set()
malicious_ip_counts = {}

log_interval = 10  # Log after every 10 batches
ban_threshold = 5
num_epochs = 10  # Number of epochs for training

# Model file paths
MODEL_FILE_PATH = 'packet_hv_model.pth'
TEXT_MODEL_FILE_PATH = 'packet_hv_text_model.pth'

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize hypervector encoders for packet features
class HypervectorEncoder:
    def __init__(self, dimension, num_levels=NUM_LEVELS):
        self.dimension = dimension
        self.num_levels = num_levels
        self.level_hvs = self._generate_level_hvs()
        self.feature_hvs = {}

    def _generate_random_hv(self):
        hv = np.random.choice([-1, 1], size=self.dimension)
        return hv

    def _generate_level_hvs(self):
        level_hvs = {}
        for level in range(self.num_levels):
            level_hvs[level] = self._generate_random_hv()
        return level_hvs

    def encode_categorical(self, feature_name, value):
        if feature_name not in self.feature_hvs:
            self.feature_hvs[feature_name] = {}
        if value not in self.feature_hvs[feature_name]:
            self.feature_hvs[feature_name][value] = self._generate_random_hv()
        return self.feature_hvs[feature_name][value]

    def encode_numerical(self, feature_name, value, min_value, max_value):
        # Quantize the value into levels
        level = int((value - min_value) / (max_value - min_value) * (self.num_levels - 1))
        return self.level_hvs[level]

    def bundle(self, vectors):
        # Element-wise sum and then normalize
        bundled = np.sum(vectors, axis=0)
        bundled = np.sign(bundled)
        return bundled

# Initialize hypervector encoders for text data
class TextHypervectorEncoder:
    def __init__(self, dimension):
        self.dimension = dimension
        self.char_hvs = self._generate_char_hvs()

    def _generate_char_hvs(self):
        char_hvs = {}
        for char in (chr(i) for i in range(32, 127)):  # ASCII characters
            char_hvs[char] = np.random.choice([-1, 1], size=self.dimension)
        return char_hvs

    def encode_text(self, text):
        hvs = []
        for char in text:
            if char in self.char_hvs:
                hvs.append(self.char_hvs[char])
            else:
                hvs.append(np.zeros(self.dimension))
        if hvs:
            bundled = np.sum(hvs, axis=0)
            bundled = np.sign(bundled)
            return bundled
        else:
            return np.zeros(self.dimension)

# Initialize hypervector encoders
packet_hv_encoder = HypervectorEncoder(HV_DIMENSION)
text_hv_encoder = TextHypervectorEncoder(HV_DIMENSION)

class CombinedHVDataset(Dataset):
    def __init__(self, hv_data, targets):
        self.hv_data = hv_data
        self.targets = targets

    def __len__(self):
        return len(self.hv_data)

    def __getitem__(self, idx):
        hv_features = self.hv_data[idx]
        target = self.targets[idx]
        return hv_features, target

class HVModel(nn.Module):
    def __init__(self, input_dim, num_categories):
        super(HVModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_categories)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def setup_logging():
    """
    Sets up logging
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler('exceptions.log')
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

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

def extract_label(explanation):
    if 'attack' in explanation.lower() or 'vulnerable' in explanation.lower():
        return 1  # Malicious
    else:
        return 0  # Benign

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

    features = {
        'ip_version': ip_version,
        'ip_len': ip_len,
        'tcp_sport': tcp_sport,
        'tcp_dport': tcp_dport,
        'tcp_flags': tcp_flags
    }
    return features

def encode_packet_features(features):
    hv_list = []

    # Encode numerical features
    hv_list.append(packet_hv_encoder.encode_numerical('ip_version', features['ip_version'], 0, 6))
    hv_list.append(packet_hv_encoder.encode_numerical('ip_len', features['ip_len'], 0, 65535))
    hv_list.append(packet_hv_encoder.encode_numerical('tcp_sport', features['tcp_sport'], 0, 65535))
    hv_list.append(packet_hv_encoder.encode_numerical('tcp_dport', features['tcp_dport'], 0, 65535))
    hv_list.append(packet_hv_encoder.encode_numerical('tcp_flags', features['tcp_flags'], 0, 255))

    # Bundle all feature hypervectors
    packet_hv = packet_hv_encoder.bundle(hv_list)
    return packet_hv

def preprocess_data(dataset):
    hv_data = []
    targets = []
    for item in dataset:
        features = extract_features(item['Packet/Tags'])
        packet_hv = encode_packet_features(features)
        hv_data.append(packet_hv)

        label = extract_label(item['Explanation'])
        targets.append(label)

    hv_data = np.stack(hv_data)
    hv_data = torch.tensor(hv_data, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    return hv_data, targets

def preprocess_text_data(dataset):
    hv_data = []
    targets = []
    for item in dataset:
        text = item['Explanation']
        text_hv = text_hv_encoder.encode_text(text)
        hv_data.append(text_hv)

        label = extract_label(item['Explanation'])
        targets.append(label)

    hv_data = np.stack(hv_data)
    hv_data = torch.tensor(hv_data, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    return hv_data, targets

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (hv_features, targets) in enumerate(train_loader):
        hv_features, targets = hv_features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(hv_features)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            logging.info(f'Train Epoch: {epoch} [{batch_idx * len(hv_features)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(train_loader)
    logging.info(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')

def evaluate(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for hv_features, targets in test_loader:
            hv_features, targets = hv_features.to(device), targets.to(device)
            outputs = model(hv_features)
            preds = outputs.argmax(dim=1, keepdim=True).squeeze()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate evaluation metrics
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)

    logging.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

def train_and_evaluate(model, device, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)
    # Save the model after training
    torch.save(model.state_dict(), MODEL_FILE_PATH)  # Save the model weights

def load_feedback_file(feedback_file_path):
    feedback_data = {}
    try:
        with open(feedback_file_path, 'r') as file:
            for line in file:
                packet_id, label = line.strip().split(',')
                feedback_data[packet_id] = int(label)
    except FileNotFoundError:
        logging.info("Feedback file not found. Creating a new feedback file.")
        open(feedback_file_path, 'w').close()  # This creates an empty file
    return feedback_data

def packet_capture(queue, interface='eth0'):
    logging.info(f"Starting packet capture on {interface}. Press Ctrl+C to stop.")
    def capture(packet):
        logging.info(f"Packet captured: {packet.summary()}")
        queue.put(packet)

    try:
        sniff(iface=interface, prn=capture, stop_filter=lambda x: shutdown_event.is_set())
    except PermissionError:
        logging.error("Error: Insufficient permissions to capture packets.")
        exit(1)

def process_and_redirect(queue, model, device, feedback_data, filter_ipv6=True, show_https=True, protocol_range=(80, 443)):
    while not shutdown_event.is_set():
        try:
            packet = queue.get(timeout=1)  # Timeout to check for shutdown event
            process_packet(packet, model, device)
        except Empty:
            continue
        except Exception as e:
            logging.exception(f"Error processing packet: {e}")

def process_packet(packet, model, device):
    try:
        packet_hv = preprocess_packet(packet)
        if packet_hv is None:
            logging.info("Packet preprocessing returned None, skipping.")
            return

        packet_hv = torch.tensor(packet_hv, dtype=torch.float32).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            output = model(packet_hv)
            prediction = torch.argmax(output, dim=1).item()

        if prediction != 0:
            redirect_ip = '192.168.1.101'  # Replace with actual IP
            redirect_packet(packet, redirect_ip)
            packet_id = hashlib.sha256(packet.build()).hexdigest()
            logging.info(f"Redirected packet {packet_id}, classified as {attack_types[prediction]}")
    except Exception as e:
        logging.exception(f"Error processing packet: {e}")

def redirect_packet(packet, analysis_server_ip):
    """Redirects the given packet to the specified analysis server IP."""
    if packet.haslayer(IP):
        redirected_packet = packet.copy()
        redirected_packet[IP].dst = analysis_server_ip
        send(redirected_packet)

def preprocess_packet(packet):
    if not packet.haslayer(IP) and not packet.haslayer(TCP) and not packet.haslayer(UDP):
        return None

    features = {}
    features['ip_version'] = packet.version if packet.haslayer(IP) else DEFAULT_IP_VERSION
    features['ip_len'] = packet.len if packet.haslayer(IP) else DEFAULT_IP_LEN
    features['tcp_sport'] = packet[TCP].sport if packet.haslayer(TCP) else DEFAULT_TCP_SPORT
    features['tcp_dport'] = packet[TCP].dport if packet.haslayer(TCP) else UNCOMMON_PORT

    tcp_flags = 0
    if packet.haslayer(TCP):
        tcp_flags = sum([packet[TCP].flags.F, packet[TCP].flags.S << 1, packet[TCP].flags.R << 2,
                         packet[TCP].flags.P << 3, packet[TCP].flags.A << 4, packet[TCP].flags.U << 5,
                         packet[TCP].flags.E << 6, packet[TCP].flags.C << 7])
    features['tcp_flags'] = tcp_flags

    src_ip = packet[IP].src if packet.haslayer(IP) else None

    if src_ip:
        if src_ip in banned_ips:
            return None  # Skip processing the packet if the IP is banned

        with malicious_ip_counts_lock:
            count = malicious_ip_counts.get(src_ip, 0) + 1
            malicious_ip_counts[src_ip] = count
            if count >= ban_threshold:
                with banned_ips_lock:
                    banned_ips.add(src_ip)
                    logging.info(f"IP {src_ip} has been banned.")
                return None  # Skip further processing for banned IPs

    packet_hv = encode_packet_features(features)
    return packet_hv

if __name__ == '__main__':
    import argparse

    setup_logging()

    feedback_data = load_feedback_file('packet_feedback.txt')
    parser = argparse.ArgumentParser(description='Packet Classifier with Hypervectors')
    parser.add_argument('--mode', type=str, choices=['train', 'capture'], required=True, help='Operation mode: train or capture')
    parser.add_argument('--interface', type=str, required=False, default='eth0', help='Network interface to capture packets from')

    args = parser.parse_args()

    if args.mode == 'train':
        # Load and preprocess the dataset
        dataset = load_dataset('rdpahalavan/packet-tag-explanation')['train']

        # Preprocess packet data
        packet_hv_data, labels = preprocess_data(dataset)

        # Split the dataset into training and testing sets for packet features
        hv_train, hv_test, labels_train, labels_test = train_test_split(
            packet_hv_data, labels, test_size=0.2, random_state=42
        )

        # Initialize CombinedHVDataset for packet data training and testing
        train_dataset = CombinedHVDataset(hv_data=hv_train, targets=labels_train)
        test_dataset = CombinedHVDataset(hv_data=hv_test, targets=labels_test)

        # Create DataLoader instances for packet data training and testing
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)

        # Initialize the model
        model = HVModel(input_dim=HV_DIMENSION, num_categories=2).to(device)

        # Train and evaluate the model
        logging.info("Training Packet Hypervector Model...")
        train_and_evaluate(model, device, train_loader, test_loader, epochs=num_epochs)

        logging.info(f'Model saved to {MODEL_FILE_PATH}')

    elif args.mode == 'capture':
        if not os.path.exists(MODEL_FILE_PATH):
            logging.error("Model file not found. Please train the model first.")
            exit()

        # Load the model
        model = HVModel(input_dim=HV_DIMENSION, num_categories=2).to(device)
        model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device))
        model.eval()

        packet_queue = Queue()

        # Start packet capture thread
        capture_thread = threading.Thread(target=packet_capture, args=(packet_queue, args.interface))
        capture_thread.start()

        # Start packet processing thread
        processing_thread = threading.Thread(target=process_and_redirect, args=(packet_queue, model, device, feedback_data))
        processing_thread.start()

        # Keep the main thread running until a keyboard interrupt is received
        try:
            while True:
                time.sleep(1)  # Sleep and let other threads do the work
        except KeyboardInterrupt:
            logging.info("Shutdown signal received. Shutting down gracefully.")
            shutdown_event.set()  # Signal threads to shut down

        # Wait for threads to complete upon receiving a shutdown signal
        capture_thread.join()
        processing_thread.join()

        logging.info("All threads have been shut down.")
