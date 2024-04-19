# Load pre-trained ResNet-50 model
resnet = models.resnet50(pretrained=True)
# Remove the last fully connected layer
resnet = nn.Sequential(*list(resnet.children())[:-1])
# Set the model to evaluation mode
resnet.eval()
# Define a function to extract features from an input image using ResNet-50
def extract_features_from_image(input_image_path, model):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(input_image_path)
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    features = torch.flatten(features)
    return features.numpy()

# Define a function to retrieve similar images based on an input image
def retrieve_similar_images_from_input_image(input_image_features, k=5):
    predicted_label = svm_classifier.predict([input_image_features])[0]
    similar_indices = [i for i, label in enumerate(labels) if label == predicted_label]
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(extracted_features[similar_indices])
    distances, indices = nn_model.kneighbors([input_image_features])
    original_indices = [similar_indices[i] for i in indices[0]]
    return original_indices

# Define a function to visualize retrieved images
def visualize_retrieved_images(input_image_path, similar_image_indices):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title('Input Image')
    plt.axis('off')

    for i, index in enumerate(similar_image_indices):
        plt.subplot(1, 6, i + 2)
        image_data = np.stack((stored_data[index][b'R'], stored_data[index][b'G'], stored_data[index][b'B']), axis=-1)
        plt.imshow(Image.fromarray(image_data))
        plt.title(f'Similar Image {i + 1}')
        plt.axis('off')

    plt.show()

# Define a function to perform image retrieval
def image_retrieval(input_image_path):
    input_image_features = extract_features_from_image(input_image_path, resnet)
    similar_image_indices = retrieve_similar_images_from_input_image(input_image_features)
    visualize_retrieved_images(input_image_path, similar_image_indices)
