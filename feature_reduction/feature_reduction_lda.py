import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#loading the extracted features
extracted_features = np.load('extracted_features.npy')

#loading the labels
labels = [stored_data[i][b'labels'] for i in range(len(stored_data))]

#initializing the LDA model
lda = LinearDiscriminantAnalysis()

#fitting the LDA model
lda.fit(extracted_features, labels)

# Transform the features to the lower-dimensional space
features_lda = lda.transform(extracted_features)
