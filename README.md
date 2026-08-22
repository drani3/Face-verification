# Face Verification using Siamese Neural Networks

A real-time face verification application built with **TensorFlow/Keras** and **Kivy** that uses Siamese neural networks to verify whether a face captured from the webcam matches known identities.

## Overview

This project implements a one-shot learning approach using Siamese neural networks with L1 distance metric for face verification. The application captures live video from your webcam and compares the input face against a set of verification images to determine if the person is verified or not.

## Features

- **Real-time Face Verification**: Live webcam feed with instant verification results
- **Siamese Neural Network**: Efficient one-shot learning for face recognition
- **Configurable Thresholds**: Adjustable detection and verification thresholds
- **Kivy GUI**: User-friendly desktop application interface
- **Visual Feedback**: Color-coded verification status (Green = Verified, Red = Unverified)

## Project Structure

```
Face-verification/
├── model.ipynb                          # Jupyter notebook with model training code
├── README.md                            # This file
├── application_data/
│   ├── faceid.py                        # Main Kivy application
│   ├── layer.py                         # Custom L1Dist layer implementation
│   ├── input_image/                     # Temporary storage for captured images
│   └── verification_images/             # Reference images for verification
├── data/
│   ├── anchor/                          # Anchor face images for training
│   ├── positive/                        # Positive examples (same person)
│   └── negative/                        # Negative examples (different people)
└── siamesemodel.h5                      # Trained Siamese model (not included - see Setup)
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Kivy
- OpenCV (cv2)
- NumPy

## Installation

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd Face-verification
```

### 2. Install Dependencies

```bash
pip install tensorflow kivy opencv-python numpy
```

### 3. Set Up the Model

Since the model file (`siamesemodel.h5`) is not included due to GitHub size restrictions:

- Train the model using the provided `model.ipynb` notebook, OR
- Place a pre-trained `siamesemodel.h5` file in the project root directory

### 4. Prepare Training Data

The project requires training data organized in the `data/` directory:

- **anchor/**: Images of the person(s) to be verified
- **positive/**: More images of the same person(s) for positive training examples
- **negative/**: 300+ images of different people for negative training examples

### 5. Set Up Verification Images

Add reference images of known identities to:
```
application_data/verification_images/
```

These images will be used to verify against the webcam input.

## Usage

### Running the Application

```bash
python application_data/faceid.py
```

### How It Works

1. **Launch the App**: Start the Kivy application
2. **Position Your Face**: Look at the webcam (the application captures a 250x250 pixel crop)
3. **Click Verify**: Press the "Verify" button to check if your face matches the verification images
4. **View Results**: 
   - **Green "Verified"**: Your face matched the known identities
   - **Red "Unverified"**: Your face did not match

### Adjusting Thresholds

Edit the thresholds in `faceid.py` (line 62-63) to fine-tune verification:

```python
detection_threshold = 0.5      # Minimum confidence per comparison
verification_threshold = 0.5   # Minimum percentage of matches required
```

## How Siamese Networks Work

This project uses a **Siamese neural network** architecture:

- **Input**: Two face images (input from webcam + verification image)
- **Shared Weights**: Both images are processed through the same neural network
- **L1 Distance**: Computes the absolute difference between embeddings
- **Output**: Distance score (lower = more similar)

The network learns to:
- Extract discriminative features from faces
- Produce similar embeddings for the same person
- Produce different embeddings for different people

## Model Architecture

The custom `L1Dist` layer (in `layer.py`) computes:
```
L1Distance = Σ|embedding1 - embedding2|
```

This metric is used to determine face similarity and make verification decisions.

## File Descriptions

| File | Purpose |
|------|---------|
| `faceid.py` | Main Kivy application with webcam integration and verification logic |
| `layer.py` | Custom TensorFlow layer implementing L1 distance metric |
| `model.ipynb` | Jupyter notebook for training the Siamese network |

## Future Improvements

- [ ] Add face detection (align faces before verification)
- [ ] Support for multiple users/identities
- [ ] Database integration for storing verified users
- [ ] Improved UI with progress indicators
- [ ] Export verification logs
- [ ] Mobile deployment using Kivy

## Troubleshooting

### "Module not found" errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`

### Model not loading
- Verify `siamesemodel.h5` exists in the project root directory
- Train a new model using `model.ipynb`

### Webcam not working
- Check camera permissions
- Ensure no other application is using the webcam
- Try changing the camera index in `faceid.py` (default: 0)

## License

This project is provided as-is for educational and research purposes.
