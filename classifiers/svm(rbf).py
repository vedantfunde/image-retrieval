from sklearn.svm import SVC
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#loading the trained SVM classifier
svm_classifier = SVC(kernel='rbf')

#fitting the SVM classifier to the training data
svm_classifier.fit(extracted_features, labels)

#making predictions on the test data using the trained classifier
predicted_labels = svm_classifier.predict(test_extracted_features)

#calculating accuracy
accuracy = np.mean(predicted_labels == test_labels)
#print("Accuracy:", accuracy)
