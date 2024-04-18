import tarfile
import pickle
import requests
from io import BytesIO
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# Load pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True)
# Remove the last fully connected layer
resnet = nn.Sequential(*list(resnet.children())[:-1])
# Set the model to evaluation mode
resnet.eval()

# Define a function to preprocess and extract features from an image
def extract_features_from_data(image_data, model):
    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image_data)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Extract features
    with torch.no_grad():
        features = model(image_tensor)
    # Remove the batch dimension
    features = torch.flatten(features)
    return features.numpy()  # Convert features to NumPy array for easier handling

# Example usage
num_images = 10000
extracted_features = np.zeros((num_images, 2048))  # Array to store extracted features

for i in range(num_images):
    # Assuming 'stored_data' contains the preprocessed image data as described in the previous code
    image_data = np.stack((stored_data[i][b'R'], stored_data[i][b'G'], stored_data[i][b'B']), axis=-1)
    features = extract_features_from_data(image_data, resnet)
    extracted_features[i] = features


# Save the extracted features to a file
np.save('extracted_features.npy', extracted_features)