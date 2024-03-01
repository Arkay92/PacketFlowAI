# PacketFlowAI
PacketFlowAI is a sophisticated tool powered by deep learning, specifically designed for the real-time classification of network packets using Convolutional Neural Networks (CNN). By integrating the robust capabilities of PyTorch for model development and Scapy for packet capture and processing, PacketFlowAI is a highly efficient solution for discerning various types of network traffic.

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
git clone https://github.com/yourusername/PacketFlowAI.git
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
For on-the-fly classification of network traffic, enter the following:

```bash
python main.py --mode capture [--interface <interface_name>]
```
The --interface flag is optional and allows you to specify the network interface for packet capture. If not provided, it defaults to 'eth0'.

Note: Ensure you possess the requisite permissions for packet capture on the chosen network interface.

## Customization

PacketFlowAI offers flexibility allowing you to tailor the CNN architecture, dataset, and training parameters. Simply adjust the settings in the main.py script to align with your specific requirements.

## Contributing

Contributions to PacketFlowAI are welcome! Please feel free to submit pull requests or open issues to suggest improvements or add new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
