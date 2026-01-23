Visual Storyteller: Image Captioning with CNN-LSTM
Deep Learning Final Project by Soso Pkhakadze, Guga Mepisashvili, and Lile Vakhtangadze
Overview
This project implements an image captioning system that generates natural language descriptions of images. The model combines a Convolutional Neural Network (CNN) encoder with a Long Short-Term Memory (LSTM) decoder to bridge visual and linguistic modalities.
Architecture

Encoder: ResNet-50 (pretrained, fine-tuned on layer4 and fc)
Decoder: Single-layer LSTM with embedding layer
Embedding Size: 256
Hidden Size: 256
Vocabulary: Built from training captions (freq_threshold=5)

Dataset

Source: Flickr8k (8,000 images)
Annotations: 5 captions per image
Location: caption_data/

Installation
bashpip install -r requirements.txt
Usage
Training
Run data_and_training.ipynb to:

Load and preprocess data
Build vocabulary
Train the model (25 epochs)
Save model weights to best_model.pth

Inference
Run inference.ipynb to:

Load trained model
Generate captions for test images
Compare generated vs. ground truth captions

Key Function:
pythongenerate_caption(model, image_path, transform, vocab, device, max_length=50)
Training Details

Optimizer: Adam (lr=3e-4)
Loss: CrossEntropyLoss (ignoring pad tokens)
Scheduler: ReduceLROnPlateau
Gradient Clipping: max_norm=1.0
Batch Size: 32

Results
The model successfully generates contextually relevant captions. Examples include accurate descriptions of objects, actions, and settings, though occasional misclassifications occur in complex scenes.
Files

data_and_training.ipynb - Training pipeline
inference.ipynb - Caption generation and evaluation
best_model.pth - Trained model weights
vocab.pkl - Saved vocabulary object
requirements.txt - Dependencies

Requirements

PyTorch
torchvision
Pillow
pandas
matplotlib