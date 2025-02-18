Lip Reading using 2D CNN for Word Level
This project demonstrates lip reading using a 2D Convolutional Neural Network (CNN) model to identify words based on lip movements. It uses computer vision and deep learning techniques to detect and classify words from video frames.
![Lip Reading Animation](https://via.placeholder.com/600x400?text=Lip+Reading+Animation)
## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [License](#license)
## Description
This project aims to assist individuals suffering from dysarthria by identifying words based on lip movements. The system leverages facial landmark detection to focus on the mouth region, which is then used as input to a deep learning model to classify the words. The model uses a 2D CNN architecture trained on a dataset of lip images.
## Installation
To run the project, follow the instructions below:
### Clone the Repository
```bash
git clone https://github.com/yourusername/Lip_Reading_Using_2D_CNN_For_Word_Level.git
cd Lip_Reading_Using_2D_CNN_For_Word_Level
```
### Install the Dependencies
Create a virtual environment (optional but recommended) and install the necessary packages.
```bash
pip install -r requirements.txt
```
### Download the Pre-trained Model Weights and Dataset
Place the model weights and dataset in the model/ folder (ensure model_weights.h5 and model.json are in the folder). Download or create the dataset with images of lip movements. The dataset should be pre-processed and saved as X.txt.npy and Y.txt.npy.
## Dependencies
The following libraries are required:
- Python 3.x
- OpenCV
- Keras
- dlib
- imutils
- numpy
- pickle
You can install the necessary dependencies using the provided requirements.txt:
```bash
pip install -r requirements.txt
```
## Usage
1. Ensure that you have the pre-trained model weights (model_weights.h5), model configuration (model.json), and dataset (X.txt.npy and Y.txt.npy) available in the model/ directory.
2. Run the script:
```bash
python lip_reading.py
```
The program will start capturing video from your webcam, detecting faces, and analyzing lip movements. If a word is identified based on lip movement, it will be displayed on the screen. Press "q" to exit the video stream.
## Model Architecture
The model is based on a 2D CNN architecture:
- Convolution Layer 1: 64 filters of size 3x3 with ReLU activation
- MaxPooling Layer 1: Pooling size 2x2
- Convolution Layer 2: 32 filters of size 3x3 with ReLU activation
- MaxPooling Layer 2: Pooling size 2x2
- Flatten Layer: Converts the 2D matrix to a 1D vector
- Dense Layer 1: Fully connected layer with 256 units and ReLU activation
- Dense Layer 2: Output layer with 10 units (for the 10 words) and softmax activation
## Training
The model was trained using the lip movement images in the dataset. It uses categorical cross-entropy loss and the Adam optimizer:
- The model was trained for 250 epochs with a batch size of 16.
- The training accuracy can be printed after loading the model and history file.
## License
This project is licensed under the MIT License - see the LICENSE file for details.
