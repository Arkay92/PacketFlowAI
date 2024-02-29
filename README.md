# PacketFlowAI
PacketFlowAI is a deep learning-based tool designed for the real-time classification of network packets using Convolutional Neural Networks (CNN). It leverages PyTorch for model training and Scapy for packet capture and processing, enabling efficient identification of various types of network traffic.

## Features
- Real-time packet capture and classification
- Custom CNN architecture for packet feature analysis
- Training and evaluation on a labeled dataset
- Normalization and scaling of packet features for optimal performance
- Easy-to-use command-line interface for training and live packet capture modes

## Installation
To set up PacketFlowAI, follow these steps:

1. Clone the repository:
```
git clone https://github.com/yourusername/PacketFlowCNN.git
```
2. Navigate to the cloned directory:
```
cd PacketFlowCNN
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

This will train the model using the dataset specified in the code and save the best-performing model for future use.

### Live Capture Mode

To classify live network traffic, use the following command:
```
python main.py --mode capture
```

Ensure you have the necessary permissions to capture packets on the specified network interface.

## Customization

You can customize the model architecture, dataset, and training parameters by modifying the `main.py` script according to your needs.

## Contributing

Contributions to PacketFlowAI are welcome! Please feel free to submit pull requests or open issues to suggest improvements or add new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
