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


#trained svm loading
svm_classifier = SVC(kernel='rbf')
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

#cross-validation
cv_scores = cross_val_score(svm_classifier, features_pca, labels, cv=kf)

print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))


#fit svm data
svm_classifier.fit(extracted_features, labels)

#predicting on the test data
predicted_labels = svm_classifier.predict(test_extracted_features)

accuracy = np.mean(predicted_labels == test_labels)
print("Accuracy:", accuracy)
