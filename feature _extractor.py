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
# Perform PCA for dimensionality reduction
from sklearn.decomposition import PCA

# Directory path
directory = 'cifar-10-batches-py'

# Download the .tar.gz file
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
response = requests.get(url)
tar_bytes = BytesIO(response.content)

# Extract the contents of the .tar.gz file
tar = tarfile.open(fileobj=tar_bytes, mode="r:gz")
tar.extractall()
tar.close()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

# Assuming the .tar.gz file contains pickled data files, you can now unpickle them
d = unpickle('cifar-10-batches-py/data_batch_1')

# Extract features using ResNet-50
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])
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
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        features = model(image_tensor)
    features = torch.flatten(features)
    return features.numpy()

# Extract features for all images
num_images = 1000
extracted_features = np.zeros((num_images, 2048))

for i in range(num_images):
    image_data = np.stack((d[b'data'][i].reshape(96,32)[0:32],
                           d[b'data'][i].reshape(96,32)[32:64],
                           d[b'data'][i].reshape(96,32)[64:96]), axis=-1)
    features = extract_features_from_data(image_data, resnet)
    extracted_features[i] = features

# Save the extracted features to a file
np.save('extracted_features.npy', extracted_features)
