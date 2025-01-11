# BCNE - Bank Card Number Extractor

BCNE (Bank Card Number Extractor) is a real-time application that detects and extracts the 16-digit number from bank cards using YOLOWorld and PaddleOCR. The model runs efficiently with TensorRT for high-speed inference.

## Features
- Real-time bank card detection and number extraction
- Optimized with TensorRT for fast inference
- Supports external webcam for a better experience
- Simple usage with a single script execution

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/mahmoudmhashem/BCNE.git
cd BCNE
```

### Step 2: Install PyTorch with CUDA
To enable GPU acceleration, install PyTorch with CUDA support by following the instructions on the official PyTorch website: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

### Step 3: Install Dependencies
Ensure you have Python installed (recommended version: 3.8 or later). Then, install the required packages:
```bash
pip install opencv-python ultralytics paddleocr tensorrt
```

### Step 4: Install PaddleOCR
Since PaddleOCR requires additional dependencies, install them separately:
```bash
pip install paddlepaddle paddleocr
```

### Step 5: Set Up TensorRT
Follow the official NVIDIA guide to install TensorRT: [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)

## Using BCNE
Run the following command to start the application:
```bash
python visa_number_recognition.py
```

[Click here to view the script](https://github.com/mahmoudmhashem/BCNE/blob/main/visa_number_recognition.py)

### External Webcam Support
For a better experience, you can connect your phone as a webcam using **Iriun Webcam**:
1. Install **Iriun Webcam** on your phone ([Android](https://play.google.com/store/apps/details?id=com.jacksoftw.webcam) / [iOS](https://apps.apple.com/app/iriun-webcam/id1505650022)).
2. Install **Iriun Webcam** on your PC from [Iriun's official site](https://iriun.com/).
3. Connect your phone and PC to the same Wi-Fi network and start the Iriun app.
4. The BCNE application will automatically detect the external webcam.

### Extracting Multiple Card Numbers
- Once the 16-digit number is extracted, the app **freezes** to allow you to review the result.
- To extract another number, **press 'U'** (Unfreeze) to continue scanning.

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contribution
Feel free to contribute by submitting issues or pull requests to improve the project!

