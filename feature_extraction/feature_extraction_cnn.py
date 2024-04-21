#UTILISING CNN TO EXTRACT FEATURES

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



resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])

#evaluating the model
resnet.eval()

def extract_features_from_data(image_data, model):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image_data)
    #adding batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    #extracting features
    with torch.no_grad():
        features = model(image_tensor)
    features = torch.flatten(features)
    return features.numpy()

num_images = 10000
extracted_features = np.zeros((num_images, 2048)) 

for i in range(num_images):
    image_data = np.stack((stored_data[i][b'R'], stored_data[i][b'G'], stored_data[i][b'B']), axis=-1)
    features = extract_features_from_data(image_data, resnet)
    extracted_features[i] = features
np.save('extracted_features.npy', extracted_features)
