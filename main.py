import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from scapy.all import sniff, IP, TCP, UDP, send
import re
from sklearn.model_selection import train_test_split
import hashlib

FEATURE_MEAN = torch.tensor([50, 0.5])  # Example mean values for each feature
FEATURE_STD = torch.tensor([20, 0.5])   # Example std dev values for each feature
MIN_VALUE = torch.tensor([0, 0])        # Minimum value of each feature in the training set
MAX_VALUE = torch.tensor([100, 1])      # Maximum value of each feature in the training set

banned_ips = set()
no_feedback_packets = set()
malicious_ip_counts = {}
ban_threshold = 5

class PacketCNN(nn.Module):
    def __init__(self):
        super(PacketCNN, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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

def port_to_feature(port):
    # Example mapping for common ports, extend as needed
    port_map = {
        'ftp': 21,
        'ssh': 22,
        'http': 80,
        'https': 443
    }
    # Return the mapped value or a default for uncommon ports
    return port_map.get(port, 9999)  # 9999 can be used as a placeholder for uncommon ports

def flags_to_feature(flags):
    # Example mapping for TCP flags
    flags_map = {
        'F': 1,  # FIN
        'S': 2,  # SYN
        'R': 3,  # RST
        'P': 4,  # PSH
        'A': 5,  # ACK
        'U': 6,  # URG
        'E': 7,  # ECE
        'C': 8   # CWR
    }
    # Sum the values of all flags present
    return sum(flags_map.get(flag, 0) for flag in flags)  # Add 0 if the flag is not in the map

def extract_features(description):
    # Set default values
    default_values = {'ip_version': 0, 'ip_len': 0, 'tcp_sport': 0, 'tcp_dport': 9999, 'tcp_flags': 0}
    
    # Extract features with regex, falling back to default values if not found
    ip_version = float(re.search(r'IP version: (\d+\.\d+)', description).group(1)) if re.search(r'IP version: (\d+\.\d+)', description) else default_values['ip_version']
    ip_len = float(re.search(r'IP len: (\d+\.\d+)', description).group(1)) if re.search(r'IP len: (\d+\.\d+)', description) else default_values['ip_len']
    tcp_sport = float(re.search(r'TCP sport: (\d+)', description).group(1)) if re.search(r'TCP sport: (\d+)', description) else default_values['tcp_sport']
    
    # Handling TCP dport with port_to_feature conversion
    tcp_dport_match = re.search(r'TCP dport: (\w+)', description)
    tcp_dport = port_to_feature(tcp_dport_match.group(1)) if tcp_dport_match else default_values['tcp_dport']
    
    # Handling TCP flags with flags_to_feature conversion
    tcp_flags_match = re.search(r'TCP flags: (\w+)', description)
    tcp_flags = flags_to_feature(tcp_flags_match.group(1)) if tcp_flags_match else default_values['tcp_flags']

    # Create the feature tensor with a fixed size
    features = [ip_version, ip_len, tcp_sport, tcp_dport, tcp_flags]
    return torch.tensor(features, dtype=torch.float32)

def extract_label(explanation):
    # Simplified approach: check for keywords indicating an attack
    if 'attack' in explanation or 'vulnerable' in explanation:
        return 1  # Indicate an attack
    else:
        return 0  # Indicate normal

def preprocess_data(dataset):
    data = []
    targets = []
    for item in dataset:
        features = extract_features(item['Packet/Tags']).unsqueeze(0)  # Add a batch dimension
        label = extract_label(item['Explanation'])
        data.append(features)
        targets.append(label)
    
    data = torch.cat(data, dim=0)  # Concatenate along the batch dimension
    targets = torch.tensor(targets, dtype=torch.long)
    return data, targets

def preprocess_packet(packet):
    if not packet.haslayer(TCP) and not packet.haslayer(UDP):
        return None
    
    # Basic features
    packet_length = len(packet)
    protocol_type = 0 if packet.haslayer(TCP) else 1  # 0 for TCP, 1 for UDP
    
    # Additional features (similar to extract_features)
    ip_version = packet.version if packet.haslayer(IP) else 0  # Fallback to 0 if IP layer is not present
    ip_len = packet.len if packet.haslayer(IP) else 0  # Fallback to 0 if IP layer is not present
    tcp_sport = packet[TCP].sport if packet.haslayer(TCP) else 0  # Fallback to 0 if TCP layer is not present
    tcp_dport = packet[TCP].dport if packet.haslayer(TCP) else 9999  # Use 9999 for uncommon ports or if TCP layer is not present
    tcp_flags = 0
    if packet.haslayer(TCP):
        # Extracting TCP flags (converting flag bits to a single value for simplicity)
        tcp_flags = sum([packet[TCP].flags.F, packet[TCP].flags.S << 1, packet[TCP].flags.R << 2, packet[TCP].flags.P << 3, packet[TCP].flags.A << 4, packet[TCP].flags.U << 5, packet[TCP].flags.E << 6, packet[TCP].flags.C << 7])
    if packet.haslayer(IP):
        src_ip = packet[IP].src
        if src_ip in banned_ips:
            return  # Drop the packet by not processing it further
    features = torch.tensor([ip_version, ip_len, tcp_sport, tcp_dport, tcp_flags], dtype=torch.float32)
    
    # Ensure the tensor has the correct shape (1, 5) for the model input
    features = features.unsqueeze(0)
    
    return features
    
def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                  f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

def evaluate(model, device, test_loader, log_interval=10):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += len(data)

            if batch_idx % log_interval == 0:
                print(f"Test: [{batch_idx * len(data)}/{len(test_loader.dataset)}"
                      f" ({100. * batch_idx / len(test_loader):.0f}%)]")

    test_loss /= total_samples
    accuracy = 100. * correct / total_samples
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({accuracy:.0f}%)\n')
    
    return test_loss, accuracy

def train_and_evaluate(model, device):
    raw_dataset = load_dataset('rdpahalavan/packet-tag-explanation')['train']
    data, targets = preprocess_data(raw_dataset)
    train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)

    train_dataset = PacketDataset(train_data, train_targets)
    test_dataset = PacketDataset(test_data, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')

    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        test_loss, _ = evaluate(model, device, test_loader)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'packet_cnn_model.pth')
            print("Model saved successfully!")

    return train_loader  # Return the train_loader for use in live packet capture and retraining

def redirect_packet(packet):
    analysis_server_ip = '192.168.1.101'  # IP address of the analysis server
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

def collect_feedback(packet_id, packet):
    feedback_file = 'packet_feedback.txt'  # File where feedback is stored
    try:
        with open(feedback_file, 'r') as file:
            for line in file:
                id, label = line.strip().split(',')
                if id == packet_id:
                    return int(label)  # Return the feedback for the given packet_id
    except FileNotFoundError:
        print("Feedback file not found.")
        with open(feedback_file, 'w') as file:
            pass  # Creating an empty file if not found
    return None  # Return None if no feedback is found for the packet_id

def process_and_redirect(packet, model, device, optimizer, train_loader, filter_ipv6=True, show_https=True, protocol_range=(80, 443)):
    def strip_port(packet, src_port, dst_port):
        packet.dport = dst_port
        packet.sport = src_port

    if packet.haslayer(IP):
        if filter_ipv6 and packet[IP].version == 6:
            print("Skipping IPV6 packet.")
            return

    features = preprocess_packet(packet)
    if features is None:
        return

    tensor_features = features.to(device)
    model.eval()
    with torch.no_grad():
        output = model(tensor_features)
        prediction = output.argmax(dim=1).item()

    # print(f"Packet processed, model prediction: {prediction}")

    if show_https:
        if packet.haslayer(TCP) and protocol_range[0] <= packet[TCP].dport <= protocol_range[1]:
            strip_port(packet[TCP], packet[TCP].dport, 80)
        else:
            print("Skipping non-HTTPS traffic.")
            return

    if prediction == 1:
        packet_id = hashlib.sha256(bytes(packet)).hexdigest()
        print(f"Redirecting bad packet with ID: {packet_id}")

        # Enhanced logging for bad packets
        if packet.haslayer(IP):
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            print(f"Source IP: {src_ip}, Destination IP: {dst_ip}")
            # Update count for the source IP
            if src_ip in malicious_ip_counts:
                malicious_ip_counts[src_ip] += 1
            else:
                malicious_ip_counts[src_ip] = 1
            
             # Check if the IP exceeds the ban threshold
            if malicious_ip_counts[src_ip] >= ban_threshold:
                print(f"IP {src_ip} has exceeded the ban threshold and will be banned.")
                # Add the IP to the banned_ips set
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

        feedback = collect_feedback(packet_id, packet)
        if feedback is not None:
            print(f"Feedback for packet ID: {packet_id} is {feedback}. Retraining model...")
            new_data = tensor_features.unsqueeze(0)
            new_labels = torch.tensor([feedback], dtype=torch.long).to(device)
            
            model.train()
            optimizer.zero_grad()
            output = model(new_data)
            loss = nn.CrossEntropyLoss()(output, new_labels)
            loss.backward()
            optimizer.step()

            test_loss, accuracy = evaluate(model, device, train_loader)
            print(f"After retraining - Loss: {test_loss}, Accuracy: {accuracy}%")
        else:
            if packet_id not in no_feedback_packets:
                print(f"No feedback available for packet ID: {packet_id}")
                no_feedback_packets.add(packet_id)

def capture_live_packets(interface, model, device, optimizer, filter_ipv6=True, show_https=True, protocol_range=(80, 443)):
    train_loader = train_and_evaluate(model, device)  # Obtain the train_loader by training and evaluating the model
    print("Starting packet capture. Press Ctrl+C to stop.")
    try:
        sniff(iface=interface, prn=lambda packet: process_and_redirect(packet, model, device, optimizer, train_loader, filter_ipv6, show_https, protocol_range))
    except KeyboardInterrupt:
        print("\nStopped packet capture.")

def parse_args():
    parser = argparse.ArgumentParser(description='Packet Classifier')
    parser.add_argument('--filter-ipv6', action='store_true', help='Filter IPV6 packets (default: True)')
    parser.add_argument('--show-https', action='store_true', help='Show only HTTPS related traffic (default: True)')
    parser.add_argument('--protocol', type=str, default='80:443', help='Protocol range to show (default: 80:443)')
    return parser.parse_args()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Packet Classifier')
    parser.add_argument('--mode', type=str, choices=['train', 'capture'], required=True, help='Operation mode: train or capture')
    parser.add_argument('--interface', type=str, required=False, default='eth0', help='Network interface to capture packets from')

    # Add optional arguments for filtering IPV6, showing HTTPS, and specifying protocol range
    parser.add_argument('--filter-ipv6', action='store_true', help='Filter IPV6 packets (default: True)')
    parser.add_argument('--show-https', action='store_true', help='Show only HTTPS related traffic (default: True)')
    parser.add_argument('--protocol', type=str, default='80:443', help='Protocol range to show (default: 80:443)')

    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PacketCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.mode == 'train':
        train_and_evaluate(model, device)
    elif args.mode == 'capture':
        protocol_range = tuple(map(int, args.protocol.split(':')))
        capture_live_packets(args.interface, model, device, optimizer, args.filter_ipv6, args.show_https, protocol_range)
