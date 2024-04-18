from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'extracted_features' contains the extracted features from images

pca = PCA(n_components=1143)  # Specify the number of components
features_pca = pca.fit_transform(extracted_features)

# Now 'features_pca' contains the reduced-dimensional features after PCA
print(features_pca.shape)  # Output: (1000, 512)

# Save the reduced-dimensional features to a file
np.save('features_pca.npy', features_pca)

explained_variance_ratio = pca.explained_variance_ratio_

# Compute cumulative explained variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Choose the smallest n_components that captures a satisfactory amount of variance (e.g., 99%)
desired_variance = 0.99
n_components_satisfactory = np.argmax(cumulative_explained_variance >= desired_variance) + 1