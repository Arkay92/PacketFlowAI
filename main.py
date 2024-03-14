import logging, time, torch, re, hashlib, os, threading, signal
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from scapy.all import sniff, IP, TCP, UDP, send
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
from queue import Queue, Empty
from threading import Lock, Event
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

# Constants for feature extraction
UNCOMMON_PORT = 9999
DEFAULT_IP_VERSION = 0
DEFAULT_IP_LEN = 0
DEFAULT_TCP_SPORT = 0
DEFAULT_TCP_DPORT = UNCOMMON_PORT
DEFAULT_TCP_FLAGS = 0

# Initialize locks for thread-safe operations
banned_ips_lock = Lock()
malicious_ip_counts_lock = Lock()

# Event for graceful shutdown
shutdown_event = Event()
attack_types = ['benign', 'DDoS', 'port_scan', 'malware', 'phishing', 'other']  # Example attack types

banned_ips = set()
no_feedback_packets = set()
malicious_ip_counts = {}

log_interval = 10  # Log after every 10 batches, adjust as per your requirement
ban_threshold = 5
num_epochs = 10  # Or any other number you find suitable for your training

# Model file path
MODEL_FILE_PATH = 'packet_cnn_model.pth'
TEXT_MODEL_FILE_PATH = 'packet_cnn_text_model.pth'

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=100) 
tfidf_vectorizer.fit(attack_types)

# Dynamically set the text feature size based on the fitted vectorizer
text_feature_size = 100  # This will be dynamically updated after fitting the vectorizer
    
class CombinedModel(nn.Module):
    def __init__(self, packet_feature_size, text_feature_size, num_categories):
        super(CombinedModel, self).__init__()
        self.packet_branch = nn.Sequential(
            nn.Linear(packet_feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.text_feature_size = text_feature_size
        combined_feature_size = 64 + self.text_feature_size  # Adjusted to ensure accuracy

        self.combined_fc = nn.Sequential(
            nn.Linear(164, 128),  # Adjusted to match the combined features size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_categories)
        )

    def forward(self, packet_features, text_features):
        packet_out = self.packet_branch(packet_features)
        # Debugging print statements
        # print(f"Packet out shape: {packet_out.shape}")
        # print(f"Text features shape: {text_features.shape}")

        combined_features = torch.cat((packet_out, text_features), dim=1)
        # print(f"Combined features shape: {combined_features.shape}")
        out = self.combined_fc(combined_features)
        return out

class CombinedDataset(Dataset):
    def __init__(self, packet_data, text_data, targets, text_feature_size=100):
        self.packet_data = packet_data
        self.text_data = text_data
        self.targets = targets
        self.text_feature_size = text_feature_size  # New attribute to store text feature size

    def __len__(self):
        return len(self.packet_data)

    def __getitem__(self, idx):
        packet_features = self.packet_data[idx]
        target = self.targets[idx]
        
        # Correctly handle text_features whether they are None or not
        text_features = self.text_data[idx] if self.text_data is not None else torch.zeros(self.text_feature_size)

        return packet_features, text_features, target

class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_categories):
        super(TextClassifier, self).__init__()
        # Define the architecture
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, num_categories)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def normalize_features(features):
    max_values = torch.tensor([1, 65535, 65535, 65535, 255], dtype=torch.float32)  # Example max values
    return features / max_values

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

# Load both Packet and Text models
def initialize_models(packet_model_file_path, text_model_file_path, device, packet_feature_size, text_feature_size, num_categories):
    packet_model = CombinedModel(packet_feature_size, text_feature_size, num_categories).to(device)
    if os.path.exists(packet_model_file_path):
        logging.info(f"Loading packet model from {packet_model_file_path}")
        packet_model.load_state_dict(torch.load(packet_model_file_path, map_location=device))
    else:
        logging.info("No packet model found. Initializing a new one.")

    text_model = TextClassifier(input_dim=text_feature_size, num_categories=num_categories).to(device)
    if os.path.exists(text_model_file_path):
        logging.info(f"Loading text model from {text_model_file_path}")
        text_model.load_state_dict(torch.load(text_model_file_path, map_location=device))
    else:
        logging.info("No text model found. Initializing a new one.")

    packet_model.eval()
    text_model.eval()

    return packet_model, text_model

def load_model(model_path, device):
    model = CombinedModel(packet_feature_size=5, text_feature_size=100, num_categories=6)  # Adjust text_feature_size if necessary
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model

def update_model_with_new_data(model, new_packet_features, new_labels, optimizer, device):
    model.train()
    new_packet_features, new_labels = new_packet_features.to(device), new_labels.to(device)
    optimizer.zero_grad()
    outputs = model(new_packet_features)
    loss = nn.CrossEntropyLoss()(outputs, new_labels)
    loss.backward()
    optimizer.step()

def prepare_text_data_for_training(dataset):
    explanations = [item['Explanation'] for item in dataset]
    attack_types = [extract_attack_type(exp) for exp in explanations]

    # Convert attack types to numerical labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(attack_types)

    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=100)
    text_features = vectorizer.fit_transform(explanations).toarray()

    # Convert to PyTorch tensors
    text_features_tensor = torch.tensor(text_features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Split dataset into training and validation sets
    text_features_train, text_features_val, labels_train, labels_val = train_test_split(
        text_features_tensor, labels_tensor, test_size=0.2, random_state=42
    )

    # Create DataLoader instances for training and validation
    train_dataset = TensorDataset(text_features_train, labels_train)
    val_dataset = TensorDataset(text_features_val, labels_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    num_categories = len(label_encoder.classes_)

    return train_loader, val_loader, vectorizer.get_feature_names_out(), num_categories

def train_text_classifier(train_loader, val_loader, input_dim, num_categories):
    text_classifier = TextClassifier(input_dim, num_categories).to(device)
    optimizer = optim.Adam(text_classifier.parameters(), lr=0.001)
    num_epochs = 10  # Set the number of epochs

    for epoch in range(num_epochs):
        text_classifier.train()
        for texts, targets in train_loader:
            texts, targets = texts.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = text_classifier(texts)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

    return text_classifier

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
    logging.info(f"Starting packet capture on {interface}. Press Ctrl+C to stop.")
    def capture(packet):
        logging.info(f"Packet captured: {packet.summary()}")
        queue.put(packet)

    try:
        sniff(iface=interface, prn=capture, stop_filter=lambda x: shutdown_event.is_set())
    except PermissionError:
        logging.error("Error: Insufficient permissions to capture packets.")
        exit(1)

def process_packets(queue, model, device, optimizer, feedback_data, filter_ipv6=True, show_https=True, protocol_range=(80, 443)):
    while not shutdown_event.is_set():
        try:
            packet = queue.get(timeout=1)  # Timeout to check for shutdown event
            process_and_redirect(packet, model, device, optimizer, feedback_data, filter_ipv6, show_https, protocol_range)
        except Empty:  # Correctly catch the Empty exception when the queue is empty
            continue
        except Exception as e:
            logging.exception(f"Error processing packet: {e}")

def shutdown_handler():
    logging.info("Shutdown signal received. Shutting down gracefully.")
    shutdown_event.set()

def preprocess_packet(packet):
    if not packet.haslayer(IP) and not packet.haslayer(TCP) and not packet.haslayer(UDP):
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
        if src_ip in banned_ips:  # Check if the source IP is already banned
            return None  # Skip processing the packet if the IP is banned
        
        with malicious_ip_counts_lock:
            count = malicious_ip_counts.get(src_ip, 0) + 1
            malicious_ip_counts[src_ip] = count
            if count >= ban_threshold:
                with banned_ips_lock:
                    banned_ips.add(src_ip)
                    logging.info(f"IP {src_ip} has been banned.")
                return None  # Skip further processing for banned IPs

    features = torch.tensor([ip_version, ip_len, tcp_sport, tcp_dport, tcp_flags], dtype=torch.float32).unsqueeze(0)
    normalized_features = normalize_features(features)
    return normalized_features

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

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    log_interval = 10  # Log after every 10 batches, adjust as per your requirement

    for batch_idx, (packet_features, text_features, targets) in enumerate(train_loader):
        packet_features, text_features, targets = packet_features.to(device), text_features.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(packet_features, text_features)
        loss = nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            logging.info(f'Train Epoch: {epoch} [{batch_idx * len(packet_features)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    logging.info(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')

def evaluate(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for packet_features, text_features, targets in test_loader:
            packet_features, text_features, targets = packet_features.to(device), text_features.to(device), targets.to(device)
            outputs = model(packet_features, text_features)
            preds = outputs.argmax(dim=1, keepdim=True).squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate evaluation metrics
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_preds)

    logging.info(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}')

def train_and_evaluate(model, device, train_loader, test_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        evaluate(model, device, test_loader)
    # Save the model after training
    torch.save(model.state_dict(), MODEL_FILE_PATH)  # Save the model weights

def update_model(model, new_data, new_labels, optimizer, device):
    # Ensure new_data and new_labels match the expected dimensions
    if new_data.dim() == 1:
        new_data = new_data.unsqueeze(0).to(device)
    if new_labels.dim() == 1:
        new_labels = new_labels.unsqueeze(0).to(device)

    model.train()
    optimizer.zero_grad()
    output = model(new_data)
    loss = nn.CrossEntropyLoss()(output, new_labels)
    loss.backward()
    optimizer.step()

def extract_text_from_packet(packet):
    """
    Extracts meaningful text from the packet payload, considering different encoding and formats.
    
    Args:
        packet: The network packet from which to extract text.
        
    Returns:
        Extracted text as a string. Returns an empty string if no meaningful text is found.
    """
    # Check for the payload layer in common protocols
    if packet.haslayer(TCP) or packet.haslayer(UDP):
        try:
            payload = packet.load  # Access the payload

            # Attempt to decode payload as utf-8 or ascii
            for encoding in ['utf-8', 'ascii']:
                try:
                    return payload.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # If decoding fails, try to extract printable characters only
            return ''.join(filter(lambda x: x in string.printable, str(payload)))

        except AttributeError:
            # No payload present
            return ""
    return ""

def redirect_packet(packet, analysis_server_ip):
    """Redirects the given packet to the specified analysis server IP."""
    if packet.haslayer(IP):
        redirected_packet = packet.copy()
        redirected_packet[IP].dst = analysis_server_ip
        send(redirected_packet)

def process_and_redirect(packet, packet_model, text_model, device, tfidf_vectorizer, attack_types, redirect_ip='192.168.1.101'):
    try:
        packet_features = preprocess_packet(packet)
        if packet_features is None:
            logging.info("Packet preprocessing returned None, skipping.")
            return

        packet_features = packet_features.unsqueeze(0).to(device)
        text_data = extract_text_from_packet(packet)
        text_features = tfidf_vectorizer.transform([text_data]).toarray()
        text_features_tensor = torch.tensor(text_features, dtype=torch.float32).to(device)

        with torch.no_grad():
            packet_output = packet_model(packet_features)
            text_output = text_model(text_features_tensor)

        combined_output = (packet_output + text_output) / 2
        prediction = torch.argmax(combined_output, dim=1).item()

        if prediction != 0:
            redirect_packet(packet, redirect_ip)
            packet_id = hashlib.sha256(packet.build()).hexdigest()
            logging.info(f"Redirected packet {packet_id}, classified as {attack_types[prediction]}")
    except Exception as e:
        logging.exception(f"Error processing packet: {e}")

def adjust_dimensions(tensor):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() > 2:
        tensor = tensor.view(tensor.size(0), -1)
    return tensor


def load_and_preprocess_dataset():
    # Load the dataset from Hugging Face
    dataset = load_dataset('rdpahalavan/packet-tag-explanation')

    # Assuming the dataset has features in 'Packet/Tags' and labels in 'Explanation'
    packet_features_list = []
    text_data_list = []
    labels_list = []
    for item in dataset['train']:
        packet_feature = extract_features(item['Packet/Tags'])
        packet_features_list.append(packet_feature)
        
        text_data = item['Explanation']  # Extract text data
        text_data_list.append(text_data)
        
        label = extract_label(item['Explanation'])
        labels_list.append(label)

    # Transform text data with TF-IDF
    tfidf_vectorizer.fit(text_data_list)
    text_features = tfidf_vectorizer.transform(text_data_list).toarray()
    text_features_tensor = torch.tensor(text_features, dtype=torch.float32)

    # Update text_feature_size based on fitted vectorizer
    global text_feature_size
    text_feature_size = text_features_tensor.shape[1]

    packet_features_tensor = torch.stack(packet_features_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    return packet_features_tensor, text_features_tensor, labels_tensor

if __name__ == '__main__':
    import argparse

    setup_logging()

    feedback_data = load_feedback_file('packet_feedback.txt')
    parser = argparse.ArgumentParser(description='Packet Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'capture'], required=True, help='Operation mode: train or capture')
    parser.add_argument('--interface', type=str, required=False, default='eth0', help='Network interface to capture packets from')
    parser.add_argument('--filter-ipv6', action='store_true', help='Filter IPV6 packets (default: True)')
    parser.add_argument('--show-https', action='store_true', help='Show only HTTPS related traffic (default: True)')
    parser.add_argument('--protocol', type=str, default='80:443', help='Protocol range to show (default: 80:443)')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize both packet and text models
    packet_model, text_model = initialize_models(MODEL_FILE_PATH, TEXT_MODEL_FILE_PATH, device, packet_feature_size=5, text_feature_size=100, num_categories=6)
    feedback_data = load_feedback_file('packet_feedback.txt')

    if args.mode == 'train':
        # Load and preprocess the dataset
        dataset = load_dataset('rdpahalavan/packet-tag-explanation')  # Ensure you load the correct dataset

        # Prepare packet data for training
        packet_features, labels = preprocess_data(dataset['train'])  # Adjust as necessary to match your dataset structure

        # Split the dataset into training and testing sets for packet features
        packet_features_train, packet_features_test, labels_train, labels_test = train_test_split(
            packet_features, labels, test_size=0.2, random_state=42
        )

        # Initialize CombinedDataset for packet data training and testing
        train_packet_dataset = CombinedDataset(packet_data=packet_features_train, text_data=None, targets=labels_train, text_feature_size=text_feature_size)
        test_packet_dataset = CombinedDataset(packet_data=packet_features_test, text_data=None, targets=labels_test, text_feature_size=text_feature_size)

        # Create DataLoader instances for packet data training and testing
        train_packet_loader = DataLoader(train_packet_dataset, batch_size=64, shuffle=True)
        test_packet_loader = DataLoader(test_packet_dataset, batch_size=64)

        # Prepare text data for training
        train_text_loader, val_text_loader, vocab, num_categories = prepare_text_data_for_training(dataset['train'])

        # Initialize the models
        packet_model = CombinedModel(packet_feature_size=5, text_feature_size=text_feature_size, num_categories=num_categories).to(device)
        text_classifier = TextClassifier(input_dim=len(vocab), num_categories=num_categories).to(device)

        # Define optimizers for both models
        packet_optimizer = optim.Adam(packet_model.parameters(), lr=0.001)
        text_optimizer = optim.Adam(text_classifier.parameters(), lr=0.001)

        # Train packet model
        logging.info("Training Packet Model...")
        train_and_evaluate(packet_model, device, train_packet_loader, test_packet_loader)

        # Train text classifier
        logging.info("Starting text model training...")
        for epoch in range(num_epochs):
            logging.info(f"Starting Epoch {epoch+1}/{num_epochs}")
            total_loss = 0
            text_classifier.train()

            for batch_idx, (texts, targets) in enumerate(train_text_loader):
                texts, targets = texts.to(device), targets.to(device)
                text_optimizer.zero_grad()
                outputs = text_classifier(texts)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                text_optimizer.step()
                total_loss += loss.item()

            # Optionally, you can add evaluation steps here for the text classifier

        # Save the trained models
        torch.save(packet_model.state_dict(), MODEL_FILE_PATH)
        
        logging.info("Text model training completed.")
        # Save the model
        torch.save(text_classifier.state_dict(), TEXT_MODEL_FILE_PATH)  # Define TEXT_MODEL_FILE_PATH as needed
        logging.info(f'Model saved to {TEXT_MODEL_FILE_PATH}')
    elif args.mode == 'capture':
        if not (os.path.exists(MODEL_FILE_PATH) and os.path.exists(TEXT_MODEL_FILE_PATH)):
            logging.error("Model files not found. Please train the models first.")
            exit()

        # Load both models
        packet_model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location=device))
        text_model.load_state_dict(torch.load(TEXT_MODEL_FILE_PATH, map_location=device))
        packet_model.eval()  # Set the packet model to evaluation mode
        text_model.eval()  # Set the text model to evaluation mode

        packet_queue = Queue()
        protocol_range = tuple(map(int, args.protocol.split(':')))

        # Start packet capture thread
        capture_thread = threading.Thread(target=packet_capture, args=(packet_queue, args.interface))
        capture_thread.start()

        # Keep the main thread running until a keyboard interrupt is received
        try:
            while True:
                time.sleep(1)  # Sleep and let other threads do the work
        except KeyboardInterrupt:
            logging.info("Shutdown signal received. Shutting down gracefully.")
            shutdown_event.set()  # Signal threads to shut down

        # Start packet processing thread, now with both models
        processing_thread = threading.Thread(target=process_and_redirect, args=(packet_queue, packet_model, text_model, device, feedback_data, args.filter_ipv6, args.show_https, protocol_range))
        processing_thread.start()

        # Wait for threads to complete upon receiving a shutdown signal
        capture_thread.join()
        processing_thread.join()

        logging.info("All threads have been shut down.")
