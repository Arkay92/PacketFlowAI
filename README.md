# PacketFlowAI
PacketFlowAI is a cutting-edge network security tool that uses Hyperdimensional Computing (HDC) to classify network packets in real-time. By transitioning from traditional Convolutional Neural Networks (CNN) to hyperdimensional computing, PacketFlowAI delivers improved robustness, scalability, and efficiency for packet classification. It leverages PyTorch for model development and Scapy for real-time packet capture and processing, providing a powerful solution for detecting and classifying various types of network traffic.

## Features
- Real-time packet capture and classification with hyperdimensional computing
- Hypervector-based data representation for robust and noise-tolerant classification
- Customizable model architecture designed for high-dimensional input
- Training and evaluation on labeled datasets using hyperdimensional techniques
- Flexible command-line interface for training and live packet capture modes
- Scalability and adaptability to new features and network threats

## Installation
To set up PacketFlowAI, follow these steps:

1. Clone the repository:
```
git clone https://github.com/Arkay92/PacketFlowAI.git
```
2. Navigate to the cloned directory:
```
cd PacketFlowAI
```

3. Install the required Python packages:
```
pip install -r requirements.txt
```

## Usage
PacketFlowAI can be run in two modes: **training mode** and **live capture mode**.

### Training Mode
To train the model on your dataset, use the following command:
```
python main.py --mode train
```

This command will preprocess the dataset, convert the packet features and textual data into hypervectors, and train the model using hyperdimensional techniques. The best-performing model is saved for future use.

### Live Capture Mode
For on-the-fly classification of network traffic, enter the following:

```bash
python main.py --mode capture [--interface <interface_name>]
```
The --interface flag allows you to specify the network interface for packet capture (e.g., eth0 or wlan0). If not provided, it defaults to eth0.

Ensure you have the necessary permissions for capturing packets on the chosen interface.

PacketFlowAI will capture packets, convert their features into hypervectors, and classify them using the trained model.

## Customization

PacketFlowAI provides flexibility, allowing you to customize the model architecture, dataset, and hyperdimensional parameters. You can adjust the settings in the main.py script to suit your specific use case, including:

- Modifying the hypervector dimensions and encoding methods.
- Altering the model architecture (e.g., number of layers, dropout rates).
- Adjusting training parameters (e.g., batch size, learning rate, number of epochs).

## Key Concepts in PacketFlowAI
- Hyperdimensional Computing: A novel approach where data is represented as high-dimensional vectors (hypervectors), which offers increased robustness to noise and scalability in complex environments.
- Binding and Permutation: Core operations in hyperdimensional computing that encode associations between data and capture sequential information.

## Contributing

Contributions to PacketFlowAI are encouraged! If you have ideas for improvement, new features, or bug fixes, please submit a pull request or open an issue. Together, we can make PacketFlowAI more robust and feature-rich.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
