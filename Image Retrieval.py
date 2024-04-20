import gdown
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC

# Download and load PCA features
url_features_pca = 'https://drive.google.com/uc?id=19mjP1Kf28k1DEVycYyoGzJHTtwjZaExM'
gdown.download(url_features_pca, 'features_pca.npy', quiet=False)
features_pca = np.load('features_pca.npy')

# Download and load labels
url_labels = 'https://drive.google.com/uc?id=1wbW1ZKNyT9bLsaIi1uOmD9HQ4OZeqDqm'
gdown.download(url_labels, 'labels.npy', quiet=False)
labels = np.load('labels.npy')

# Download and load stored data
url_stored_data = 'https://drive.google.com/uc?id=1pim6rKyVM8K_nGPNtPkI3dwh7_zcSzJY'
gdown.download(url_stored_data, 'stored_data.pkl', quiet=False)
with open('stored_data.pkl', 'rb') as f:
    stored_data = pickle.load(f)

# Train SVM classifier
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(features_pca, labels)

# Load pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

# Define function to extract features from an image
def extract_features_from_image(input_image, model, pca):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(input_image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    features = torch.flatten(features)
    features = pca.transform(features.unsqueeze(0))
    return features[0]

# Define function to retrieve similar images from an input image
def retrieve_similar_images_from_input_image(input_image_features, k=5):
    predicted_label = svm_classifier.predict([input_image_features])[0]
    similar_indices = [i for i, label in enumerate(labels) if label == predicted_label]
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(features_pca[similar_indices])
    distances, indices = nn_model.kneighbors([input_image_features])
    original_indices = [similar_indices[i] for i in indices[0]]
    return original_indices

# Define function to visualize retrieved images
def visualize_retrieved_images(input_image, similar_image_indices):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')

    for i, index in enumerate(similar_image_indices):
        plt.subplot(1, 6, i + 2)
        image_data = np.stack((stored_data[index][b'R'], stored_data[index][b'G'], stored_data[index][b'B']), axis=-1)
        plt.imshow(Image.fromarray(image_data))
        plt.title(f'Similar Image {i + 1}')
        plt.axis('off')

    plt.show()

# Define main image retrieval function
def image_retrieval(input_image_path):
    input_image = Image.open(input_image_path)
    input_image_features = extract_features_from_image(input_image, resnet, pca)
    similar_image_indices = retrieve_similar_images_from_input_image(input_image_features)
    visualize_retrieved_images(input_image, similar_image_indices)



# Example usage
# input_image_path = '/content/drive/MyDrive/ml/project/dog-puppy-on-garden-royalty-free-image-1586966191.jpg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/download (1).jpeg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/download (2).jpeg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/download (3).jpeg'
# image_retrieval(input_image_path)

# input_image_path = '/content/drive/MyDrive/ml/project/download (5).jpeg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/download (6).jpeg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/download (7).jpeg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/download (8).jpeg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/pexels-helena-lopes-1996332.jpg'
# image_retrieval(input_image_path)
# input_image_path = '/content/drive/MyDrive/ml/project/istockphoto-1496285190-170667a.webp'
# image_retrieval(input_image_path)
