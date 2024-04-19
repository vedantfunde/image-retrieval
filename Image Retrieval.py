import gdown
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from io import BytesIO  # Import BytesIO module

# Download the extracted features file
print(extracted_features.shape)

# Load pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True)
# Remove the last fully connected layer
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# Set the model to evaluation mode
resnet.eval()

# Define functions for image retrieval
def extract_features_from_image(input_image, model):
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
    return features.numpy()

def retrieve_similar_images_from_input_image(input_image_features, k=5):
    predicted_label = svm_classifier.predict([input_image_features])[0]
    similar_indices = [i for i, label in enumerate(labels) if label == predicted_label]
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(extracted_features[similar_indices])
    distances, indices = nn_model.kneighbors([input_image_features])
    original_indices = [similar_indices[i] for i in indices[0]]
    return original_indices

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

def image_retrieval(input_image_url):
    # Fetch the input image
    response = requests.get(input_image_url)
    input_image = Image.open(BytesIO(response.content))
    
    # Extract features from the input image
    input_image_features = extract_features_from_image(input_image, resnet)
    
    # Retrieve similar images
    similar_image_indices = retrieve_similar_images_from_input_image(input_image_features)
    
    # Visualize retrieved images
    visualize_retrieved_images(input_image, similar_image_indices)

# Example usage
input_image_url = 'https://drive.google.com/uc?id=1-LCCrr8zQGP19Zyn52IvmEAVGiJIdH0D'
image_retrieval(input_image_url)
