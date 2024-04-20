import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def extract_features_from_image(input_image, model, lda):
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
    features = lda.transform(features.unsqueeze(0))
    return features[0]

def retrieve_similar_images_from_input_image(input_image_features, k=5):
    predicted_label = svm_classifier.predict([input_image_features])[0]
    similar_indices = [i for i, label in enumerate(labels) if label == predicted_label]
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(features_lda[similar_indices])
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

def image_retrieval(input_image_path):
    input_image = Image.open(input_image_path)
    input_image_features = extract_features_from_image(input_image, resnet, lda)
    similar_image_indices = retrieve_similar_images_from_input_image(input_image_features)
    visualize_retrieved_images(input_image, similar_image_indices)

# input_image_path = '/content/drive/MyDrive/ml/project/dog-puppy-on-garden-royalty-free-image-1586966191.jpg'
# image_retrieval(input_image_path)
