import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the extracted features
extracted_features = np.load('extracted_features.npy')

# Load the labels
labels = [stored_data[i][b'labels'] for i in range(len(stored_data))]

# Initialize the LDA model
lda = LinearDiscriminantAnalysis()

# Fit the LDA model to the data
lda.fit(extracted_features, labels)

# Compute explained variance ratio for each component
explained_variance_ratio = lda.explained_variance_ratio_

# Compute cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

extracted_features = np.load('extracted_features.npy')

# Load the labels
labels = [stored_data[i][b'labels'] for i in range(len(stored_data))]

# Initialize the LDA model
lda = LinearDiscriminantAnalysis(n_components=9)  # You can adjust the number of components as needed

# Fit the LDA model to the data
lda.fit(extracted_features, labels)

# Transform the features to the lower-dimensional space
features_lda = lda.transform(extracted_features)