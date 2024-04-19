from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

pca = PCA(n_components=1143)  # Specify the number of components
features_pca = pca.fit_transform(extracted_features)
#print(features_pca.shape) 

#saving the reduced-dimensional features to a file
np.save('features_pca.npy', features_pca)
